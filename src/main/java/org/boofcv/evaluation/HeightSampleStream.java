package org.boofcv.evaluation;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A recorded takeoff-relative height stream, replayed <b>causally</b>.
 *
 * <p>The file is a recording of a sensor, not a table the estimator may index into. This class
 * enforces that distinction structurally: {@link #drainUpTo} hands out only the samples that had
 * become available at or before a given frame time, in order, once each. A caller cannot reach a
 * later sample even by accident, so an offline replay is causally identical to a live stream and
 * {@code DEC-VO-007} D5's guarantee holds in both.
 *
 * <p>Latency is applied here rather than in the channel, because latency is a property of the
 * transport a caller is modelling and not of the zero-order hold: a sample taken at {@code t} is
 * published with {@code availabilityTime = t + latency}.
 */
public final class HeightSampleStream {

    /** One recorded sample: when it became usable, and what it said. */
    public record Sample(double availabilityTimeS, double relativeHeightM) {}

    private final List<Sample> samples;
    private int cursor;

    public HeightSampleStream(List<Sample> samples) {
        this.samples = List.copyOf(samples);
        for (int i = 1; i < this.samples.size(); i++) {
            if (this.samples.get(i).availabilityTimeS() < this.samples.get(i - 1).availabilityTimeS()) {
                throw new IllegalArgumentException(
                        "height samples must be in non-decreasing availability order; row " + i
                        + " is " + this.samples.get(i).availabilityTimeS() + " after "
                        + this.samples.get(i - 1).availabilityTimeS());
            }
        }
    }

    /**
     * Reads a height stream from a CSV.
     *
     * @param csv          path to the file; must have a header row
     * @param timeColumn   column holding the sample instant, seconds
     * @param valueColumn  column holding takeoff-relative height, metres
     * @param latencyS     added to every sample instant to give its availability time
     * @param decimation   keep every {@code decimation}-th row (1 = all)
     */
    public static HeightSampleStream fromCsv(Path csv, String timeColumn, String valueColumn,
                                             double latencyS, int decimation) throws IOException {
        return fromCsv(csv, timeColumn, valueColumn, latencyS, decimation, false);
    }

    /**
     * @param zeroAtFirstSample subtract the first sample's value from every sample, turning an
     *                          ABSOLUTE height column into a relative one. Used only for the
     *                          diagnostic oracle arm ({@code MetricReadoutSettings.heightSemantics}
     *                          = {@code oracle_agl_diagnostic}); a production-like takeoff-relative
     *                          channel is already zero at its datum and must not be re-zeroed.
     */
    public static HeightSampleStream fromCsv(Path csv, String timeColumn, String valueColumn,
                                             double latencyS, int decimation,
                                             boolean zeroAtFirstSample) throws IOException {
        if (decimation < 1) {
            throw new IllegalArgumentException("height_decimation must be >= 1; got " + decimation);
        }
        List<String> lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            throw new IOException("height CSV is empty: " + csv);
        }
        String[] header = lines.get(0).split(",", -1);
        int tIdx = indexOf(header, timeColumn, csv);
        int vIdx = indexOf(header, valueColumn, csv);

        List<Sample> out = new ArrayList<>();
        int kept = 0;
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            if (line.isBlank()) {
                continue;
            }
            if ((kept++ % decimation) != 0) {
                continue;
            }
            String[] f = line.split(",", -1);
            out.add(new Sample(Double.parseDouble(f[tIdx].trim()) + latencyS,
                               Double.parseDouble(f[vIdx].trim())));
        }
        if (out.isEmpty()) {
            throw new IOException("height CSV has no data rows: " + csv);
        }
        if (zeroAtFirstSample) {
            double datum = out.get(0).relativeHeightM();
            for (int i = 0; i < out.size(); i++) {
                out.set(i, new Sample(out.get(i).availabilityTimeS(),
                                      out.get(i).relativeHeightM() - datum));
            }
        }
        return new HeightSampleStream(out);
    }

    private static int indexOf(String[] header, String name, Path csv) throws IOException {
        for (int i = 0; i < header.length; i++) {
            if (header[i].trim().equals(name)) {
                return i;
            }
        }
        throw new IOException("column '" + name + "' not found in " + csv
                + " (header: " + String.join(",", header) + ")");
    }

    /**
     * Every not-yet-delivered sample whose availability time is at or before {@code frameTimeS},
     * in order. Advances the cursor, so each sample is delivered exactly once and no sample from
     * the frame's future can ever be returned.
     */
    public List<Sample> drainUpTo(double frameTimeS) {
        List<Sample> due = new ArrayList<>();
        while (cursor < samples.size() && samples.get(cursor).availabilityTimeS() <= frameTimeS) {
            due.add(samples.get(cursor++));
        }
        return due;
    }

    /** Total samples in the recording. */
    public int size() {
        return samples.size();
    }

    /** How many samples have been delivered so far. */
    public int delivered() {
        return cursor;
    }
}
