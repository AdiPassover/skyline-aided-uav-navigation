package org.boofcv.confidence;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.annotation.Nullable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.HexFormat;

/**
 * The versioned artifact mapping signals to a verdict — schema <b>2.0.0</b>, the
 * {@code DEC-CONF-002}-as-amended form (contracts/calibration-config.md).
 *
 * <p>Schema 2.0.0 replaces 1.0.0's single weighted-sum form with a discriminated
 * {@code score_model} union ({@code weighted_sum} | {@code logistic} | {@code isotonic}), and adds
 * three blocks the author's amendments require:
 *
 * <ul>
 *   <li>{@code configuration_binding} (amendment A3) — the estimator/acquisition configuration this
 *       calibration was fitted under. Scoring a run whose capture context does not match is
 *       <b>refused</b>, mirroring FR-017's refusal semantics; transfer outside the binding is
 *       unvalidated by definition.</li>
 *   <li>{@code temporal} (amendment A4) — the window {@code W} and warm-up {@code m} the windowed
 *       signals were captured under. Part of the identity: a different {@code W} is a different
 *       calibration.</li>
 *   <li>{@code probability_semantics} (amendment A1) — {@code false} unless the A1 dual gate
 *       passed. The loader enforces the structural half (it must be {@code false} while
 *       {@code validated} is false); the statistical half lives in {@code EXP-CONF-001}.</li>
 * </ul>
 *
 * <p><b>1.0.0 configurations are refused.</b> They predate {@code DEC-CONF-002} and carry neither
 * binding nor temporal identity; silently accepting one would score a run under a calibration with
 * no stated validity domain.
 *
 * <p>The digest is SHA-256 over the file's raw UTF-8 bytes with line endings normalised to
 * {@code \n}. It deliberately hashes bytes rather than a canonical re-serialisation: Java and
 * Python disagree on the textual form of some doubles (e.g. {@code 1.0E-7} vs {@code 1e-07}), so a
 * canonical-re-serialisation digest would be a latent cross-language mismatch.
 *
 * <h2>No transcendental functions on the shared path</h2>
 *
 * <p>The logistic model's score is the linear predictor pushed through the <b>algebraic squash</b>
 * {@code s(z) = 0.5 + 0.5 * (z / (1 + |z|))} rather than the logistic sigmoid. {@code exp} is not
 * required to be identically rounded across Java's {@code Math.exp} and Python's libm, which would
 * break the bitwise agreement invariant (contracts/agreement.md); the squash uses only
 * exactly-rounded IEEE-754 operations, is strictly monotone in {@code z}, and is therefore
 * order-equivalent to the sigmoid for every rank-based analysis. The true-sigmoid probability
 * mapping, if the A1 gate ever admits one, is an offline-only analysis artifact.
 */
public final class CalibrationConfig {

    /** The only schema this loader accepts. */
    public static final String SCHEMA_VERSION = "2.0.0";

    /** Comparison operators available to a rejection rule. */
    public enum Op {
        /** Strictly less than. */
        LT("lt"),
        /** Strictly greater than. */
        GT("gt");

        private final String wireName;

        Op(String wireName) {
            this.wireName = wireName;
        }

        public String wireName() {
            return wireName;
        }

        static Op fromWireName(String name) {
            for (Op o : values()) {
                if (o.wireName.equals(name)) {
                    return o;
                }
            }
            throw new IllegalArgumentException("Unknown comparison op: '" + name + "'");
        }

        /**
         * Applies the comparison.
         *
         * <p>Strictness matters and is fixed here: {@code lt} is {@code <}, never {@code <=}. A
         * value sitting exactly on a threshold must be classified identically by the Java and
         * Python implementations, and {@code <} versus {@code <=} is the classic way two
         * implementations of "the same rule" quietly disagree.
         */
        public boolean test(double value, double threshold) {
            return this == LT ? value < threshold : value > threshold;
        }
    }

    /** Normalisations available to a weighted-sum score term. */
    public enum Normalisation {
        /** Value clamped to {@code [0,1]}. Requires no reference. */
        IDENTITY("identity"),
        /** {@code min(value / reference, 1)}; higher input is better. Requires a reference. */
        SATURATING("saturating"),
        /** {@code 1 / (1 + value / reference)}; lower input is better. Requires a reference. */
        INVERSE_SATURATING("inverse_saturating");

        private final String wireName;

        Normalisation(String wireName) {
            this.wireName = wireName;
        }

        public String wireName() {
            return wireName;
        }

        static Normalisation fromWireName(String name) {
            for (Normalisation n : values()) {
                if (n.wireName.equals(name)) {
                    return n;
                }
            }
            throw new IllegalArgumentException("Unknown normalisation: '" + name + "'");
        }

        boolean requiresReference() {
            return this != IDENTITY;
        }

        /**
         * Normalises a present value to {@code [0,1]}.
         *
         * <p>Absent values never reach here — the scorer short-circuits to
         * {@link ConfidenceReason#SIGNALS_UNAVAILABLE} first. That ordering is the single most
         * important rule in this class: a "reasonable default" for an absent signal is exactly the
         * mechanism by which an unmeasured frame would become a confident one.
         */
        double apply(double value, @Nullable Double reference) {
            return switch (this) {
                case IDENTITY -> clamp01(value);
                case SATURATING -> clamp01(value / Objects.requireNonNull(reference));
                case INVERSE_SATURATING -> clamp01(1.0 / (1.0 + value / Objects.requireNonNull(reference)));
            };
        }
    }

    /** One ordered rejection rule: if {@code signal op threshold}, reject with {@code reason}. */
    public record RejectionRule(String signal, Op op, double threshold, ConfidenceReason reason) {
        public RejectionRule {
            Objects.requireNonNull(signal, "signal");
            Objects.requireNonNull(op, "op");
            Objects.requireNonNull(reason, "reason");
            if (!reason.isRejectionCondition()) {
                throw new IllegalArgumentException(
                        "Rejection rule reason must name a condition, got " + reason);
            }
            if (Double.isNaN(threshold) || Double.isInfinite(threshold)) {
                throw new IllegalArgumentException("threshold must be finite, got " + threshold);
            }
        }
    }

    /** One weighted score term. */
    public record ScoreTerm(String signal, Normalisation normalise, @Nullable Double reference, double weight) {
        public ScoreTerm {
            Objects.requireNonNull(signal, "signal");
            Objects.requireNonNull(normalise, "normalise");
            if (normalise.requiresReference() && reference == null) {
                throw new IllegalArgumentException(
                        "normalisation '" + normalise.wireName() + "' requires a reference for signal '" + signal + "'");
            }
            if (reference != null && (reference == 0.0 || Double.isNaN(reference) || Double.isInfinite(reference))) {
                throw new IllegalArgumentException("reference must be finite and non-zero, got " + reference);
            }
            if (Double.isNaN(weight) || Double.isInfinite(weight) || weight < 0.0) {
                throw new IllegalArgumentException("weight must be finite and >= 0, got " + weight);
            }
        }
    }

    /** One logistic coefficient over a raw signal value. */
    public record Coefficient(String signal, double coefficient) {
        public Coefficient {
            Objects.requireNonNull(signal, "signal");
            if (Double.isNaN(coefficient) || Double.isInfinite(coefficient)) {
                throw new IllegalArgumentException("coefficient must be finite, got " + coefficient);
            }
        }
    }

    /**
     * The score model — a closed union. Evaluation lives in {@link ConfidenceScorer}; this is
     * data only.
     */
    public sealed interface ScoreModel permits WeightedSum, Logistic, Isotonic {
        /** The signals the model reads, in evaluation order. */
        List<String> signals();

        String typeName();
    }

    /** Schema-1.0.0-era weighted sum of normalised terms, retained as the bootstrap plumbing form. */
    public record WeightedSum(List<ScoreTerm> terms) implements ScoreModel {
        private static final double WEIGHT_SUM_TOLERANCE = 1e-9;

        public WeightedSum {
            terms = List.copyOf(terms);
            if (terms.isEmpty()) {
                throw new IllegalArgumentException("weighted_sum terms must not be empty");
            }
            double weightSum = 0.0;
            for (ScoreTerm t : terms) {
                weightSum += t.weight();
            }
            // Not a mathematical necessity -- but a set that does not sum to 1 almost always means
            // a term was edited and another forgotten, and the score would silently leave [0,1].
            if (Math.abs(weightSum - 1.0) > WEIGHT_SUM_TOLERANCE) {
                throw new IllegalArgumentException("weighted_sum weights must sum to 1.0, got " + weightSum);
            }
        }

        @Override
        public List<String> signals() {
            List<String> out = new ArrayList<>();
            for (ScoreTerm t : terms) {
                out.add(t.signal());
            }
            return out;
        }

        @Override
        public String typeName() {
            return "weighted_sum";
        }
    }

    /**
     * Logistic model over raw signal values: {@code z = intercept + Σ cᵢ·xᵢ} accumulated
     * left-to-right in configured order, squashed algebraically (class javadoc). Coefficients are
     * fitted offline (P3+) and land here as data; nothing in this class fits anything.
     */
    public record Logistic(double intercept, List<Coefficient> coefficients) implements ScoreModel {
        public Logistic {
            coefficients = List.copyOf(coefficients);
            if (Double.isNaN(intercept) || Double.isInfinite(intercept)) {
                throw new IllegalArgumentException("intercept must be finite, got " + intercept);
            }
            if (coefficients.isEmpty()) {
                throw new IllegalArgumentException("logistic coefficients must not be empty");
            }
        }

        @Override
        public List<String> signals() {
            List<String> out = new ArrayList<>();
            for (Coefficient c : coefficients) {
                out.add(c.signal());
            }
            return out;
        }

        @Override
        public String typeName() {
            return "logistic";
        }
    }

    /**
     * Piecewise-constant monotone map over one signal — the single-index fallback
     * ({@code DEC-CONF-002} layer 1). {@code thresholds} strictly ascending;
     * {@code values.length == thresholds.length + 1}, each in {@code [0,1]}. The score is
     * {@code values[i]} where {@code i} is the number of thresholds the value is strictly greater
     * than — comparisons only, so bitwise-identical across languages by construction.
     */
    public record Isotonic(String signal, double[] thresholds, double[] values) implements ScoreModel {
        public Isotonic {
            Objects.requireNonNull(signal, "signal");
            thresholds = thresholds.clone();
            values = values.clone();
            if (values.length != thresholds.length + 1) {
                throw new IllegalArgumentException("isotonic values must have thresholds+1 entries, got "
                        + values.length + " for " + thresholds.length + " thresholds");
            }
            for (int i = 0; i < thresholds.length; i++) {
                if (Double.isNaN(thresholds[i]) || Double.isInfinite(thresholds[i])) {
                    throw new IllegalArgumentException("isotonic thresholds must be finite");
                }
                if (i > 0 && thresholds[i] <= thresholds[i - 1]) {
                    throw new IllegalArgumentException("isotonic thresholds must be strictly ascending");
                }
            }
            for (double v : values) {
                if (Double.isNaN(v) || v < 0.0 || v > 1.0) {
                    throw new IllegalArgumentException("isotonic values must be in [0,1], got " + v);
                }
            }
        }

        @Override
        public double[] thresholds() {
            return thresholds.clone();
        }

        @Override
        public double[] values() {
            return values.clone();
        }

        @Override
        public List<String> signals() {
            return List.of(signal);
        }

        @Override
        public String typeName() {
            return "isotonic";
        }
    }

    private final String schemaVersion;
    private final String calibrationId;
    private final boolean validated;
    @Nullable
    private final String validatedBy;
    private final boolean probabilitySemantics;
    private final List<String> requiredSignals;
    private final Map<String, Object> configurationBinding;
    private final int windowW;
    private final int warmupM;
    private final List<RejectionRule> rejectionRules;
    private final ScoreModel scoreModel;
    private final double usableScoreBound;
    private final String digest;

    private CalibrationConfig(Dto dto, String digest) {
        this.schemaVersion = require(dto.schemaVersion, "schema_version");
        if (!SCHEMA_VERSION.equals(schemaVersion)) {
            throw new IllegalArgumentException("Unsupported calibration schema_version '" + schemaVersion
                    + "'; this loader accepts " + SCHEMA_VERSION + " only. 1.0.0 configurations predate "
                    + "DEC-CONF-002 and carry no configuration binding; re-author them under 2.0.0.");
        }
        this.calibrationId = require(dto.calibrationId, "calibration_id");
        this.validated = requireBoolean(dto.validated, "validated");
        this.validatedBy = dto.validatedBy;
        this.digest = digest;

        this.probabilitySemantics = requireBoolean(dto.probabilitySemantics, "probability_semantics");
        if (probabilitySemantics && (!validated || validatedBy == null)) {
            // Amendment A1's structural half: a probability claim requires a recorded validation.
            throw new IllegalArgumentException("probability_semantics may be true only for a validated "
                    + "calibration naming its validating experiment (amendment A1)");
        }

        this.requiredSignals = List.copyOf(require(dto.requiredSignals, "required_signals"));
        for (String s : requiredSignals) {
            assertKnownSignal(s);
        }

        Map<String, Object> binding = require(dto.configurationBinding, "configuration_binding");
        if (binding.isEmpty()) {
            throw new IllegalArgumentException("configuration_binding must not be empty (amendment A3): "
                    + "a calibration with no stated validity domain cannot refuse a mismatched run");
        }
        for (Map.Entry<String, Object> e : binding.entrySet()) {
            Object v = e.getValue();
            if (!(v instanceof Number || v instanceof String || v instanceof Boolean)) {
                throw new IllegalArgumentException("configuration_binding values must be scalar; '"
                        + e.getKey() + "' is " + (v == null ? "null" : v.getClass().getSimpleName()));
            }
        }
        this.configurationBinding = java.util.Collections.unmodifiableMap(new LinkedHashMap<>(binding));

        Dto.TemporalDto temporal = require(dto.temporal, "temporal");
        this.windowW = (int) requireDouble(toDouble(require(temporal.windowW, "temporal.window_w")), "temporal.window_w");
        this.warmupM = (int) requireDouble(toDouble(require(temporal.warmupM, "temporal.warmup_m")), "temporal.warmup_m");
        if (windowW < 2 || warmupM < 2 || warmupM > windowW) {
            throw new IllegalArgumentException("temporal requires 2 <= warmup_m <= window_w; got W="
                    + windowW + ", m=" + warmupM);
        }

        List<RejectionRule> rules = new ArrayList<>();
        for (Dto.RuleDto r : require(dto.rejectionRules, "rejection_rules")) {
            String signal = require(r.signal, "rejection_rules[].signal");
            assertKnownSignal(signal);
            rules.add(new RejectionRule(
                    signal,
                    Op.fromWireName(require(r.op, "rejection_rules[].op")),
                    requireDouble(r.threshold, "rejection_rules[].threshold"),
                    ConfidenceReason.fromWireName(require(r.reason, "rejection_rules[].reason"))));
        }
        this.rejectionRules = List.copyOf(rules);

        this.scoreModel = parseScoreModel(require(dto.scoreModel, "score_model"));

        this.usableScoreBound = requireDouble(dto.usableScoreBound, "usable_score_bound");
        if (usableScoreBound < 0.0 || usableScoreBound > 1.0) {
            throw new IllegalArgumentException(
                    "usable_score_bound must be in [0,1], got " + usableScoreBound);
        }

        // A rule or model term reading a signal the calibration never declared would make
        // required_signals -- and therefore FR-017's refusal -- silently incomplete.
        Set<String> declared = new LinkedHashSet<>(requiredSignals);
        for (RejectionRule r : rejectionRules) {
            if (!declared.contains(r.signal())) {
                throw new IllegalArgumentException(
                        "rejection rule reads '" + r.signal() + "' which is not in required_signals");
            }
        }
        for (String s : scoreModel.signals()) {
            if (!declared.contains(s)) {
                throw new IllegalArgumentException(
                        "score model reads '" + s + "' which is not in required_signals");
            }
        }
    }

    private static ScoreModel parseScoreModel(Dto.ScoreModelDto dto) {
        String type = require(dto.type, "score_model.type");
        switch (type) {
            case "weighted_sum": {
                List<ScoreTerm> terms = new ArrayList<>();
                for (Dto.TermDto t : require(dto.terms, "score_model.terms")) {
                    String signal = require(t.signal, "score_model.terms[].signal");
                    assertKnownSignal(signal);
                    terms.add(new ScoreTerm(
                            signal,
                            Normalisation.fromWireName(require(t.normalise, "score_model.terms[].normalise")),
                            t.reference,
                            requireDouble(t.weight, "score_model.terms[].weight")));
                }
                refuse(dto.intercept != null, "score_model.intercept is not a weighted_sum field");
                refuse(dto.coefficients != null, "score_model.coefficients is not a weighted_sum field");
                refuse(dto.signal != null || dto.thresholds != null || dto.values != null,
                        "isotonic fields are not weighted_sum fields");
                return new WeightedSum(terms);
            }
            case "logistic": {
                List<Coefficient> coefficients = new ArrayList<>();
                for (Dto.CoefficientDto c : require(dto.coefficients, "score_model.coefficients")) {
                    String signal = require(c.signal, "score_model.coefficients[].signal");
                    assertKnownSignal(signal);
                    coefficients.add(new Coefficient(signal,
                            requireDouble(c.coefficient, "score_model.coefficients[].coefficient")));
                }
                refuse(dto.terms != null, "score_model.terms is not a logistic field");
                refuse(dto.signal != null || dto.thresholds != null || dto.values != null,
                        "isotonic fields are not logistic fields");
                return new Logistic(requireDouble(dto.intercept, "score_model.intercept"), coefficients);
            }
            case "isotonic": {
                String signal = require(dto.signal, "score_model.signal");
                assertKnownSignal(signal);
                List<Double> thresholdsList = require(dto.thresholds, "score_model.thresholds");
                List<Double> valuesList = require(dto.values, "score_model.values");
                refuse(dto.terms != null, "score_model.terms is not an isotonic field");
                refuse(dto.intercept != null || dto.coefficients != null,
                        "logistic fields are not isotonic fields");
                double[] thresholds = new double[thresholdsList.size()];
                for (int i = 0; i < thresholds.length; i++) {
                    thresholds[i] = requireDouble(thresholdsList.get(i), "score_model.thresholds[" + i + "]");
                }
                double[] values = new double[valuesList.size()];
                for (int i = 0; i < values.length; i++) {
                    values[i] = require(valuesList.get(i), "score_model.values[" + i + "]");
                }
                return new Isotonic(signal, thresholds, values);
            }
            default:
                throw new IllegalArgumentException("Unknown score_model.type: '" + type + "'");
        }
    }

    private static void refuse(boolean condition, String message) {
        if (condition) {
            throw new IllegalArgumentException(message);
        }
    }

    /** Loads and validates a calibration from disk, computing its digest from the bytes read. */
    public static CalibrationConfig load(Path path) throws IOException {
        byte[] raw = Files.readAllBytes(path);
        String text = new String(raw, StandardCharsets.UTF_8);
        return parse(text);
    }

    /**
     * Parses and validates a calibration from JSON text.
     *
     * <p>Exposed for tests and for the offline path; production loading goes through
     * {@link #load(Path)}.
     */
    public static CalibrationConfig parse(String json) {
        ObjectMapper mapper = new ObjectMapper();
        // Refuse unknown fields: a typo in a signal name must be an error, not a silently ignored
        // term that changes the score without changing the digest's meaning to a reader.
        mapper.enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        Dto dto;
        try {
            dto = mapper.readValue(json, Dto.class);
        } catch (IOException e) {
            throw new IllegalArgumentException("Malformed calibration configuration: " + e.getMessage(), e);
        }
        return new CalibrationConfig(dto, digestOf(json));
    }

    /**
     * SHA-256 over the configuration's UTF-8 bytes with line endings normalised to {@code \n}.
     *
     * <p>See the class javadoc for why this hashes bytes rather than a re-serialisation.
     */
    static String digestOf(String json) {
        String normalised = json.replace("\r\n", "\n").replace("\r", "\n");
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(normalised.getBytes(StandardCharsets.UTF_8));
            return "sha256:" + HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    /**
     * Refuses a capture/run context that does not satisfy this calibration's binding
     * (amendment A3).
     *
     * <p>Every binding key must be present in {@code context} with an equal value — numbers
     * compared as doubles exactly, strings and booleans by equality. Absence of a bound key is a
     * refusal, never a pass: an unknown context cannot be shown to match.
     *
     * @param context     the run's capture context (manifest {@code estimator_config} plus the
     *                    confidence block's {@code capture_context})
     * @param runIdentity human-readable name of the run, for the error message
     */
    public void checkBinding(Map<String, Object> context, String runIdentity) {
        for (Map.Entry<String, Object> bound : configurationBinding.entrySet()) {
            Object actual = context.get(bound.getKey());
            if (actual == null) {
                throw new IllegalArgumentException("Calibration '" + calibrationId + "' binds '"
                        + bound.getKey() + "' but " + runIdentity + " does not carry it; "
                        + "transfer outside the binding is unvalidated (amendment A3)");
            }
            if (!bindingValueEquals(bound.getValue(), actual)) {
                throw new IllegalArgumentException("Calibration '" + calibrationId + "' requires "
                        + bound.getKey() + " = " + bound.getValue() + " but " + runIdentity
                        + " has " + actual + "; transfer outside the binding is unvalidated (amendment A3)");
            }
        }
    }

    private static boolean bindingValueEquals(Object bound, Object actual) {
        if (bound instanceof Number bn && actual instanceof Number an) {
            return bn.doubleValue() == an.doubleValue();
        }
        return bound.equals(actual);
    }

    public String schemaVersion() {
        return schemaVersion;
    }

    /** Human-readable identity, carried into every result this calibration produces. */
    public String calibrationId() {
        return calibrationId;
    }

    /**
     * Whether an experiment supports this calibration's values.
     *
     * <p>{@code false} for everything currently in the repository. Rendered into outputs, not
     * merely stored: SC-009 forbids any shipped artifact presenting an unvalidated threshold as
     * validated.
     */
    public boolean validated() {
        return validated;
    }

    /** The experiment ID that validated this calibration, or null. */
    @Nullable
    public String validatedBy() {
        return validatedBy;
    }

    /**
     * Whether the score may be read as a calibrated probability. {@code false} until the A1 dual
     * gate — adequacy floor and held-out reliability — passes, and structurally impossible to set
     * while {@code validated} is false.
     */
    public boolean probabilitySemantics() {
        return probabilitySemantics;
    }

    /** Signals this calibration reads. Drives FR-017's refusal when a run lacks one. */
    public List<String> requiredSignals() {
        return requiredSignals;
    }

    /** The A3 validity domain: capture-context facts this calibration is bound to. */
    public Map<String, Object> configurationBinding() {
        return configurationBinding;
    }

    /** Window {@code W} the windowed signals were captured under. Part of the identity. */
    public int windowW() {
        return windowW;
    }

    /** Warm-up {@code m} the windowed signals were captured under. Part of the identity. */
    public int warmupM() {
        return warmupM;
    }

    /** Ordered; first match wins, so the reason code is deterministic when several conditions hold. */
    public List<RejectionRule> rejectionRules() {
        return rejectionRules;
    }

    /** The score model. Order within it is part of the floating-point result (contracts/agreement.md). */
    public ScoreModel scoreModel() {
        return scoreModel;
    }

    public double usableScoreBound() {
        return usableScoreBound;
    }

    /** Content digest, proving two results came from byte-identical configuration. */
    public String digest() {
        return digest;
    }

    static double clamp01(double v) {
        if (v < 0.0) {
            return 0.0;
        }
        return Math.min(v, 1.0);
    }

    private static void assertKnownSignal(String name) {
        // Round-trips through SignalBlock's lookup, which throws on an unknown name. Validating at
        // load time means a typo fails when the configuration is read, not silently at frame 4000.
        SignalBlock.UNAVAILABLE.signal(name);
    }

    private static <T> T require(@Nullable T value, String field) {
        if (value == null) {
            throw new IllegalArgumentException("Missing required field: " + field);
        }
        return value;
    }

    private static boolean requireBoolean(@Nullable Boolean value, String field) {
        return require(value, field);
    }

    private static double requireDouble(@Nullable Double value, String field) {
        double v = require(value, field);
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            throw new IllegalArgumentException(field + " must be finite, got " + v);
        }
        return v;
    }

    @Nullable
    private static Double toDouble(@Nullable Integer v) {
        return v == null ? null : (double) v;
    }

    /** Jackson binding target. Kept separate so the public type can validate in its constructor. */
    private static final class Dto {
        @JsonProperty("schema_version")
        String schemaVersion;
        @JsonProperty("calibration_id")
        String calibrationId;
        @JsonProperty("validated")
        Boolean validated;
        @JsonProperty("validated_by")
        String validatedBy;
        @JsonProperty("probability_semantics")
        Boolean probabilitySemantics;
        @JsonProperty("notes")
        String notes;
        @JsonProperty("required_signals")
        List<String> requiredSignals;
        @JsonProperty("configuration_binding")
        LinkedHashMap<String, Object> configurationBinding;
        @JsonProperty("temporal")
        TemporalDto temporal;
        @JsonProperty("rejection_rules")
        List<RuleDto> rejectionRules;
        @JsonProperty("score_model")
        ScoreModelDto scoreModel;
        @JsonProperty("usable_score_bound")
        Double usableScoreBound;

        static final class TemporalDto {
            @JsonProperty("window_w")
            Integer windowW;
            @JsonProperty("warmup_m")
            Integer warmupM;
        }

        static final class RuleDto {
            @JsonProperty("signal")
            String signal;
            @JsonProperty("op")
            String op;
            @JsonProperty("threshold")
            Double threshold;
            @JsonProperty("reason")
            String reason;
        }

        static final class ScoreModelDto {
            @JsonProperty("type")
            String type;
            // weighted_sum
            @JsonProperty("terms")
            List<TermDto> terms;
            // logistic
            @JsonProperty("intercept")
            Double intercept;
            @JsonProperty("coefficients")
            List<CoefficientDto> coefficients;
            // isotonic
            @JsonProperty("signal")
            String signal;
            @JsonProperty("thresholds")
            List<Double> thresholds;
            @JsonProperty("values")
            List<Double> values;
        }

        static final class TermDto {
            @JsonProperty("signal")
            String signal;
            @JsonProperty("normalise")
            String normalise;
            @JsonProperty("reference")
            Double reference;
            @JsonProperty("weight")
            Double weight;
        }

        static final class CoefficientDto {
            @JsonProperty("signal")
            String signal;
            @JsonProperty("coefficient")
            Double coefficient;
        }
    }
}
