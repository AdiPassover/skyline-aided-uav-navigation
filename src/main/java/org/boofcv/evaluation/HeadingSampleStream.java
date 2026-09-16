package org.boofcv.evaluation;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * A recorded external heading stream, replayed <b>causally</b> — {@link HeightSampleStream}'s twin.
 *
 * <p>The file is a recording of a sensor, not a table the estimator may index into.
 * {@link #drainUpTo} hands out only the samples that had become available at or before a given frame
 * time, in order, once each, so an offline replay is causally identical to a live stream.
 *
 * <p>The two committed feeds are {@code datasets/<id>/attitude.csv}'s {@code yaw_compass_deg} (the
 * MARS-LVIG flight controller's airframe heading, ~101 Hz) and {@code groundtruth.csv}'s
 * {@code heading_deg} on a UE capture (already the nadir camera's). Which one a config points at is
 * declared by {@code heading_readout.heading_semantics}, never inferred.
 */
public final class HeadingSampleStream {

    /** One recorded sample: when it became usable, and what heading it reported. */
    public record Sample(double availabilityTimeS, double headingDeg) {}

    private final List<Sample> samples;
    private int cursor;

    public HeadingSampleStream(List<Sample> samples) {
        this.samples = List.copyOf(samples);
        for (int i = 1; i < this.samples.size(); i++) {
            if (this.samples.get(i).availabilityTimeS() < this.samples.get(i - 1).availabilityTimeS()) {
                throw new IllegalArgumentException(
                        "heading samples must be in non-decreasing availability order; row " + i
                        + " is " + this.samples.get(i).availabilityTimeS() + " after "
                        + this.samples.get(i - 1).availabilityTimeS());
            }
        }
    }

    /**
     * Reads a heading stream from a CSV.
     *
     * @param csv          path to the file; must have a header row
     * @param timeColumn   column holding the sample instant, seconds, on the frame clock
     * @param valueColumn  column holding the heading, degrees clockwise from North
     * @param latencyS     added to every sample instant to give its availability time
     * @param decimation   keep every {@code decimation}-th row (1 = all)
     */
    public static HeadingSampleStream fromCsv(Path csv, String timeColumn, String valueColumn,
                                              double latencyS, int decimation) throws IOException {
        if (decimation < 1) {
            throw new IllegalArgumentException("heading_decimation must be >= 1; got " + decimation);
        }
        List<String> lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            throw new IOException("heading CSV is empty: " + csv);
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
            String raw = f[vIdx].trim();
            if (raw.isEmpty()) {
                // An empty heading cell is an absent measurement, not a zero. Dropping the row lets
                // the channel's own staleness policy describe the outage instead of inventing North.
                continue;
            }
            out.add(new Sample(Double.parseDouble(f[tIdx].trim()) + latencyS,
                               Double.parseDouble(raw)));
        }
        if (out.isEmpty()) {
            throw new IOException("heading CSV has no usable data rows: " + csv);
        }
        return new HeadingSampleStream(out);
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
     * Every not-yet-delivered sample whose availability time is at or before {@code frameTimeS}, in
     * order. Advances the cursor, so each sample is delivered exactly once and no sample from the
     * frame's future can ever be returned.
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
