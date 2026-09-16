package org.boofcv.relocalization;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The file-based SKY → INT boundary: strict loading of both schemas, frame keying, per-view
 * validity, the honest representation of a missing West, and the sync-residual tolerance (test J).
 * T1.
 */
public class SkylineProfileFileTest {

    private static String join(double[] p) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < p.length; i++) {
            if (i > 0) b.append(';');
            b.append(p[i]);
        }
        return b.toString();
    }

    private static RelocalizationConfig config() {
        RelocalizationConfig c = new RelocalizationConfig();
        c.descriptorLength = TestProfiles.N;
        c.degenerateStdFloor = 1e-3;
        c.skylineSyncToleranceS = 0.05;
        return c;
    }

    @Test
    public void schema1LoadsValidInvalidAndDegenerateRowsAsNorthOnlyAndKeysByFrame(@TempDir Path dir) throws IOException {
        double[] flat = new double[TestProfiles.N];
        Path csv = dir.resolve(SkylineProfileFile.CSV_NAME);
        Files.writeString(csv, SkylineProfileFile.HEADER_V1 + "\n"
                + "3,0.3,obs_003,true,," + join(TestProfiles.profile(0, 0)) + "\n"
                + "7,0.7,obs_007,false,extraction refused,\n"
                + "9,0.9,obs_009,true,," + join(flat) + "\n"
                + "12,1.2,,true,," + join(TestProfiles.profile(1, 0)) + "\n");
        Files.writeString(dir.resolve(SkylineProfileFile.MANIFEST_NAME), "{\"schema_version\":\"1.0.0\",\"n_valid\":2}");

        SkylineProfileFile f = SkylineProfileFile.load(csv, config());
        assertEquals(1, f.schemaVersion());
        assertEquals(4, f.size());
        assertEquals(2, f.validCount());
        assertEquals(0, f.westAvailableCount(), "schema 1 carries no West: represented as unavailable");
        assertEquals(0.0, f.maxAbsSyncDtS(), 0.0);
        assertNotNull(f.manifest());
        assertEquals("1.0.0", f.manifest().get("schema_version"));

        SkylineObservation a = f.lookup(3, 30.0);
        assertTrue(a.valid());
        assertFalse(a.westAvailable(), "not fabricated");
        assertNull(a.syncDtS());
        assertEquals(30.0, a.timestampS(), 0.0, "the VO frame's timestamp is used");
        assertEquals("obs_003", a.observationId());
        assertEquals(1.0, a.northDescriptor().ncc(TestProfiles.descriptor(0, 0)), 1e-12);

        SkylineObservation b = f.lookup(7, 70.0);
        assertFalse(b.valid());
        assertEquals("extraction refused", b.invalidReason());

        SkylineObservation c = f.lookup(9, 90.0);
        assertFalse(c.valid(), "a flat profile is degenerate, not scored");
        assertTrue(c.invalidReason().startsWith("degenerate"));

        assertNull(f.lookup(12, 1.2).observationId());
        assertNull(f.lookup(5, 0.5), "no row → no observation");
        assertEquals(List.of(12), f.framesOutside(10));
        assertEquals(3, f.matchedCount(10));
    }

    @Test
    public void schema2LoadsDualRowsWithPerViewValidityAndTheSyncResidual(@TempDir Path dir) throws IOException {
        Path csv = dir.resolve(SkylineProfileFile.CSV_NAME);
        String n0 = join(TestProfiles.profile(0, 0)), w0 = join(TestProfiles.profile(4, 0));
        Files.writeString(csv, SkylineProfileFile.HEADER_V2 + "\n"
                // both views, paired by time 17 ms early
                + "3,0.283,-0.017,n_003,true,," + n0 + ",w_003,true,true,," + w0 + "\n"
                // North valid, West captured but refused
                + "7,0.7,0.0,n_007,true,," + n0 + ",w_007,true,false,sky mask hole,\n"
                // North valid, no West captured at all
                + "9,0.9,,n_009,true,," + n0 + ",,false,,,\n"
                // North refused, West valid: still not eligible (North is the required view)
                + "12,1.2,0.01,n_012,false,extraction refused,,w_012,true,true,," + w0 + "\n"
                // paired 80 ms off: beyond the 50 ms tolerance → invalid, explicitly
                + "15,1.58,0.08,n_015,true,," + n0 + ",w_015,true,true,," + w0 + "\n");

        SkylineProfileFile f = SkylineProfileFile.load(csv, config());
        assertEquals(2, f.schemaVersion());
        assertEquals(5, f.size());
        assertEquals(3, f.validCount(), "North-valid rows: 3, 7, 9 (15 is beyond the tolerance)");
        assertEquals(4, f.westAvailableCount());
        assertEquals(2, f.westValidCount(), "12 (North invalid) and 3; 15 is beyond the tolerance");
        assertEquals(1, f.syncRejectedCount());
        assertEquals(0.08, f.maxAbsSyncDtS(), 1e-12);

        SkylineObservation dual = f.lookup(3, 0.3);
        assertTrue(dual.northValid());
        assertTrue(dual.westAvailable());
        assertTrue(dual.westValid());
        assertEquals(-0.017, dual.syncDtS(), 1e-12);
        assertEquals("n_003", dual.observationId());
        assertEquals("w_003", dual.west().observationId());
        assertEquals(1.0, dual.westDescriptor().ncc(TestProfiles.descriptor(4, 0)), 1e-12);
        assertTrue(dual.westDescriptor().ncc(dual.northDescriptor()) < 0.9, "West is not a copy of North");

        SkylineObservation refusedWest = f.lookup(7, 0.7);
        assertTrue(refusedWest.northValid());
        assertTrue(refusedWest.westAvailable());
        assertFalse(refusedWest.westValid());
        assertEquals("sky mask hole", refusedWest.westUnusableReason());

        SkylineObservation noWest = f.lookup(9, 0.9);
        assertTrue(noWest.northValid());
        assertFalse(noWest.westAvailable());
        assertEquals("unavailable", noWest.westUnusableReason());
        assertNull(noWest.syncDtS());

        SkylineObservation northBad = f.lookup(12, 1.2);
        assertFalse(northBad.valid(), "North is the required view");
        assertTrue(northBad.westValid(), "the West view is kept, honestly, for the record");

        SkylineObservation late = f.lookup(15, 1.5);
        assertFalse(late.valid());
        assertTrue(late.invalidReason().contains("sync residual"), late.invalidReason());
        assertFalse(late.westValid(), "the whole observation is mis-paired, both views");
    }

    /** The capture identity the exporter declares is read, exposed, and never mixed with a residual. */
    @Test
    public void theDeclaredCaptureIdentityIsReadAndAnExactFileRefusesAnyResidual(@TempDir Path dir) throws IOException {
        String n0 = join(TestProfiles.profile(0, 0)), w0 = join(TestProfiles.profile(4, 0));
        String dualRow = "27,2.7,0.0,n_027,true,," + n0 + ",w_027,true,true,," + w0 + "\n";
        String manifest = "{\"schema_version\":\"2.1.0\",\"frame_pairing\":{\"mechanism\":\"sim_vo_frame_id_exact\","
                + "\"identity\":\"exact\",\"n_unsynchronized\":0}}";

        Path exact = dir.resolve("exact");
        Files.createDirectories(exact);
        Files.writeString(exact.resolve(SkylineProfileFile.CSV_NAME), SkylineProfileFile.HEADER_V2 + "\n" + dualRow);
        Files.writeString(exact.resolve(SkylineProfileFile.MANIFEST_NAME), manifest);
        SkylineProfileFile f = SkylineProfileFile.load(exact.resolve(SkylineProfileFile.CSV_NAME), config());
        assertEquals(SkylineProfileFile.IDENTITY_EXACT, f.pairingIdentity());
        assertEquals("sim_vo_frame_id_exact", f.pairingMechanism());
        assertTrue(f.exactCaptureIdentity());
        assertEquals(0.0, f.lookup(27, 2.7).syncDtS(), 0.0, "the verified residual is written, as 0");

        // A residual under an exact identity is a mixed file: refused, not tolerated.
        Path mixed = dir.resolve("mixed");
        Files.createDirectories(mixed);
        Files.writeString(mixed.resolve(SkylineProfileFile.CSV_NAME), SkylineProfileFile.HEADER_V2 + "\n" + dualRow
                + "41,4.117,0.017,n_041,true,," + n0 + ",w_041,true,true,," + w0 + "\n");
        Files.writeString(mixed.resolve(SkylineProfileFile.MANIFEST_NAME), manifest);
        IOException e = assertThrows(IOException.class,
                () -> SkylineProfileFile.load(mixed.resolve(SkylineProfileFile.CSV_NAME), config()));
        assertTrue(e.getMessage().contains("never mixed"), e.getMessage());

        // ... and so is an exact row with no residual column value at all.
        Path blank = dir.resolve("blank");
        Files.createDirectories(blank);
        Files.writeString(blank.resolve(SkylineProfileFile.CSV_NAME), SkylineProfileFile.HEADER_V2
                + "\n27,2.7,,n_027,true,," + n0 + ",,false,,,\n");
        Files.writeString(blank.resolve(SkylineProfileFile.MANIFEST_NAME), manifest);
        assertThrows(IOException.class, () -> SkylineProfileFile.load(blank.resolve(SkylineProfileFile.CSV_NAME), config()));

        // The legacy time pairing is labelled approximate — also for a 2.0 manifest without the field.
        Path legacy = dir.resolve("legacy");
        Files.createDirectories(legacy);
        Files.writeString(legacy.resolve(SkylineProfileFile.CSV_NAME), SkylineProfileFile.HEADER_V2
                + "\n41,4.117,0.017,n_041,true,," + n0 + ",w_041,true,true,," + w0 + "\n");
        Files.writeString(legacy.resolve(SkylineProfileFile.MANIFEST_NAME),
                "{\"schema_version\":\"2.0.0\",\"frame_pairing\":{\"mechanism\":\"vo_frames_nearest_time\"}}");
        SkylineProfileFile l = SkylineProfileFile.load(legacy.resolve(SkylineProfileFile.CSV_NAME), config());
        assertEquals(SkylineProfileFile.IDENTITY_APPROXIMATE, l.pairingIdentity());
        assertFalse(l.exactCaptureIdentity());
        assertEquals(0.017, l.lookup(41, 4.1).syncDtS(), 1e-12);

        // No manifest at all: unknown, never assumed exact.
        Path bare = dir.resolve("bare");
        Files.createDirectories(bare);
        Files.writeString(bare.resolve(SkylineProfileFile.CSV_NAME), SkylineProfileFile.HEADER_V2 + "\n" + dualRow);
        assertEquals(SkylineProfileFile.IDENTITY_UNKNOWN,
                SkylineProfileFile.load(bare.resolve(SkylineProfileFile.CSV_NAME), config()).pairingIdentity());

        // An identity the contract does not define is refused.
        Path odd = dir.resolve("odd");
        Files.createDirectories(odd);
        Files.writeString(odd.resolve(SkylineProfileFile.CSV_NAME), SkylineProfileFile.HEADER_V2 + "\n" + dualRow);
        Files.writeString(odd.resolve(SkylineProfileFile.MANIFEST_NAME),
                "{\"frame_pairing\":{\"mechanism\":\"x\",\"identity\":\"probably\"}}");
        assertThrows(IOException.class, () -> SkylineProfileFile.load(odd.resolve(SkylineProfileFile.CSV_NAME), config()));
    }

    @Test
    public void malformedFilesAreRefusedNotRepaired(@TempDir Path dir) throws IOException {
        RelocalizationConfig c = config();
        Path badHeader = dir.resolve("a.csv");
        Files.writeString(badHeader, "frame,ts,id,valid,reason,profile\n");
        assertThrows(IOException.class, () -> SkylineProfileFile.load(badHeader, c));

        Path dup = dir.resolve("b.csv");
        String row = "3,0.3,x,true,," + join(TestProfiles.profile(0, 0)) + "\n";
        Files.writeString(dup, SkylineProfileFile.HEADER_V1 + "\n" + row + row);
        assertThrows(IOException.class, () -> SkylineProfileFile.load(dup, c));

        Path shortProfile = dir.resolve("c.csv");
        Files.writeString(shortProfile, SkylineProfileFile.HEADER_V1 + "\n3,0.3,x,true,,1;2;3\n");
        assertThrows(IllegalArgumentException.class, () -> SkylineProfileFile.load(shortProfile, c));

        Path emptyValid = dir.resolve("d.csv");
        Files.writeString(emptyValid, SkylineProfileFile.HEADER_V1 + "\n3,0.3,x,true,,\n");
        assertThrows(IOException.class, () -> SkylineProfileFile.load(emptyValid, c));

        // Schema 2: a row that says no West was captured but carries West fields is a lie, refused.
        Path phantomWest = dir.resolve("e.csv");
        Files.writeString(phantomWest, SkylineProfileFile.HEADER_V2 + "\n3,0.3,,n,true,,"
                + join(TestProfiles.profile(0, 0)) + ",w,false,true,," + join(TestProfiles.profile(4, 0)) + "\n");
        assertThrows(IOException.class, () -> SkylineProfileFile.load(phantomWest, c));

        Path wrongWidth = dir.resolve("f.csv");
        Files.writeString(wrongWidth, SkylineProfileFile.HEADER_V2 + "\n3,0.3,,n,true,,"
                + join(TestProfiles.profile(0, 0)) + "\n");
        assertThrows(IOException.class, () -> SkylineProfileFile.load(wrongWidth, c));
    }
}
