package org.boofcv.evaluation;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.net.InetAddress;
import java.net.UnknownHostException;

/**
 * Runtime environment captured into {@link RunManifest#environment}, per
 * contracts/run-record.md. {@code isTargetHardware} defaults to {@code false}
 * (FR-049) so runtime figures are never presented as onboard performance unless
 * explicitly overridden.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonPropertyOrder({"hostname", "os", "cpu_model", "jvm_version", "heap_max_mb", "is_target_hardware"})
public class Environment {

    @JsonProperty("hostname")
    public String hostname;

    @JsonProperty("os")
    public String os;

    @JsonProperty("cpu_model")
    public String cpuModel;

    @JsonProperty("jvm_version")
    public String jvmVersion;

    @JsonProperty("heap_max_mb")
    public Integer heapMaxMb;

    @JsonProperty("is_target_hardware")
    public boolean isTargetHardware = false;

    public Environment() {
        // Jackson
    }

    public Environment(String hostname, String os, String cpuModel, String jvmVersion,
                        Integer heapMaxMb, boolean isTargetHardware) {
        this.hostname = hostname;
        this.os = os;
        this.cpuModel = cpuModel;
        this.jvmVersion = jvmVersion;
        this.heapMaxMb = heapMaxMb;
        this.isTargetHardware = isTargetHardware;
    }

    /**
     * Captures the actual development-machine environment. {@code isTargetHardware}
     * is always false here; a target-hardware run must set it explicitly.
     */
    public static Environment captureCurrent() {
        String hostnameValue;
        try {
            hostnameValue = InetAddress.getLocalHost().getHostName();
        } catch (UnknownHostException e) {
            hostnameValue = "unknown";
        }
        String osValue = System.getProperty("os.name", "unknown") + " " + System.getProperty("os.version", "");
        String cpuModelValue = System.getenv("PROCESSOR_IDENTIFIER");
        if (cpuModelValue == null) {
            cpuModelValue = "unknown";
        }
        String jvmVersionValue = System.getProperty("java.version", "unknown");
        int heapMaxMbValue = (int) (Runtime.getRuntime().maxMemory() / (1024 * 1024));
        return new Environment(hostnameValue, osValue, cpuModelValue, jvmVersionValue, heapMaxMbValue, false);
    }
}
