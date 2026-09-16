package org.boofcv.relocalization;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Python ↔ Java parity for the C1 bounded-lag matcher and the C0 baseline ({@code DEC-INT-002}).
 *
 * <p>The fixture {@code /relocalization/c1_parity_fixture.json} was produced by the <b>actual</b>
 * {@code hsreloc.matchers} implementation ({@code evaluation/tools/int/make_c1_parity_fixture.py})
 * on deterministic synthetic profiles. Java must reproduce every pair's score within 1e‑9 and its
 * winning lag, overlap and acceptance exactly, and the same ranking for the database case. A
 * hand-computed NCC is not a substitute for this. T1 — it establishes that the two
 * implementations are the same function, nothing about skylines.
 */
public class C1ParityTest {

    private static final double TOL = 1e-9;
    private static JsonNode fixture;
    private static Map<String, SkylineDescriptor> descriptors;
    private static Map<String, SkylineMatcher> matchers;

    @BeforeAll
    static void load() throws Exception {
        try (InputStream in = C1ParityTest.class.getResourceAsStream("/relocalization/c1_parity_fixture.json")) {
            assertNotNull(in, "fixture missing — run evaluation/tools/int/make_c1_parity_fixture.py");
            fixture = new ObjectMapper().readTree(in);
        }
        int n = fixture.get("n_samples").asInt();
        descriptors = new HashMap<>();
        fixture.get("profiles").fields().forEachRemaining(e -> {
            double[] p = new double[e.getValue().size()];
            for (int i = 0; i < p.length; i++) {
                p[i] = e.getValue().get(i).asDouble();
            }
            SkylineDescriptor d = SkylineDescriptor.of(p, n, 0.0);
            assertNotNull(d, e.getKey());
            descriptors.put(e.getKey(), d);
        });
        matchers = new HashMap<>();
        fixture.get("matchers").fields().forEachRemaining(e -> {
            JsonNode m = e.getValue();
            String variant = m.get("variant").asText();
            matchers.put(e.getKey(), RelocalizationConfig.MATCHER_C0.equals(variant)
                    ? SkylineMatcher.frozenNcc()
                    : SkylineMatcher.boundedLagNcc(m.get("max_lag_samples").asInt(),
                            m.get("min_overlap_frac").asDouble()));
        });
    }

    @Test
    public void everyPairReproducesThePythonScoreLagOverlapAndAcceptance() {
        int checked = 0;
        for (JsonNode pair : fixture.get("pairs")) {
            String key = pair.get("matcher").asText() + " " + pair.get("query").asText()
                    + " vs " + pair.get("reference").asText();
            MatchResult m = matchers.get(pair.get("matcher").asText())
                    .match(descriptors.get(pair.get("query").asText()),
                            descriptors.get(pair.get("reference").asText()));
            boolean accepted = pair.get("accepted").asBoolean();
            assertEquals(accepted, m.accepted(), key);
            if (accepted) {
                assertEquals(pair.get("score").asDouble(), m.score(), TOL, key + " score");
                assertEquals((int) pair.get("shift").asDouble(), m.shiftSamples(), key + " shift");
                assertEquals(pair.get("overlap").asInt(), m.overlap(), key + " overlap");
                assertEquals(pair.get("overlap_frac").asDouble(), m.overlapFraction(), TOL, key + " overlap_frac");
                JsonNode z = pair.get("score_at_zero_lag");
                if (z != null && !z.isNull()) {
                    assertNotNull(m.scoreAtZeroLag(), key + " score_at_zero_lag");
                    assertEquals(z.asDouble(), m.scoreAtZeroLag(), TOL, key + " score_at_zero_lag");
                }
                JsonNode searched = pair.get("n_alignments_searched");
                if (searched != null && !searched.isNull() && searched.asInt() > 0) {
                    assertEquals(searched.asInt(), m.alignmentsSearched(), key + " searched");
                }
            }
            checked++;
        }
        assertTrue(checked >= 100, "fixture unexpectedly small: " + checked);
    }

    @Test
    public void rankingsMatchThePythonOrderUnderTheSameTieBreak() {
        for (JsonNode ranking : fixture.get("rankings")) {
            SkylineMatcher matcher = matchers.get(ranking.get("matcher").asText());
            SkylineDescriptor q = descriptors.get(ranking.get("query").asText());
            List<String> refs = new ArrayList<>();
            ranking.get("references").forEach(r -> refs.add(r.asText()));

            record Scored(int index, String name, double score, int shift) {
            }
            List<Scored> scored = new ArrayList<>();
            for (int i = 0; i < refs.size(); i++) {
                MatchResult m = matcher.match(q, descriptors.get(refs.get(i)));
                if (m.accepted()) {
                    scored.add(new Scored(i, refs.get(i), m.score(), m.shiftSamples()));
                }
            }
            scored.sort(Comparator.comparingDouble((Scored s) -> -s.score).thenComparingInt(s -> s.index));

            JsonNode expected = ranking.get("expected_order");
            assertEquals(expected.size(), scored.size(), ranking.get("matcher").asText() + " admissible count");
            for (int i = 0; i < expected.size(); i++) {
                assertEquals(expected.get(i).asText(), scored.get(i).name,
                        ranking.get("matcher").asText() + " rank " + (i + 1));
                assertEquals(ranking.get("expected_scores").get(i).asDouble(), scored.get(i).score, TOL);
                assertEquals((int) ranking.get("expected_shifts").get(i).asDouble(), scored.get(i).shift);
            }
        }
    }

    @Test
    public void theRecordedLagBoundConversionMatchesTheDeclaredC1Bound() {
        JsonNode ex = fixture.get("lag_bound_example");
        // lag_samples_for_degrees: ceil(|deg| · n / fov) — 11.25° over 90° in 256 samples is 32,
        // the C1-32 bound the SKY lane evaluated and the config default.
        int expected = (int) Math.ceil(Math.abs(ex.get("max_lag_deg").asDouble())
                * ex.get("n_samples").asDouble() / ex.get("fov_deg").asDouble());
        assertEquals(ex.get("lag_samples").asInt(), expected);
        assertEquals(new RelocalizationConfig().maxLagSamples, expected);
    }
}
