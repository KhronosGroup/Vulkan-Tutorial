#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <memory>
#include <array>
#include <map>
#include <sstream>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

struct Stats {
    double min;
    double max;
    double mean;
    double p50;
    double p90;
    double p99;
};

Stats calculate_stats(std::vector<double>& values) {
    if (values.empty()) return {0, 0, 0, 0, 0, 0};
    std::sort(values.begin(), values.end());
    
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();
    
    return {
        values.front(),
        values.back(),
        mean,
        values[static_cast<size_t>(values.size() * 0.50)],
        values[static_cast<size_t>(values.size() * 0.90)],
        values[static_cast<size_t>(values.size() * 0.99)]
    };
}

std::string exec_vitals(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

struct RunResult {
    double latency_ms;
    long max_rss_kb;
    int exit_code;
};

RunResult run_command(const std::string& command) {
    auto start = std::chrono::high_resolution_clock::now();
    
    pid_t pid = fork();
    if (pid == 0) {
        // Child: redirect stdout/stderr to /dev/null
        FILE* dev_null = fopen("/dev/null", "w");
        dup2(fileno(dev_null), STDOUT_FILENO);
        dup2(fileno(dev_null), STDERR_FILENO);
        fclose(dev_null);
        
        execl("/bin/sh", "sh", "-c", command.c_str(), (char*)NULL);
        _exit(1);
    } else if (pid > 0) {
        int status;
        struct rusage usage;
        wait4(pid, &status, 0, &usage);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> diff = end - start;
        return {diff.count(), usage.ru_maxrss, WIFEXITED(status) ? WEXITSTATUS(status) : -1};
    }
    
    return {-1, -1, -1};
}

int main(int argc, char** argv) {
    std::string cmd = "";
    int warmup = 5;
    int iters = 50;
    std::string out_file = "vitals.csv";
    bool use_sysfs = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cmd" && i + 1 < argc) cmd = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_file = argv[++i];
        else if (arg == "--no-sysfs") use_sysfs = false;
    }

    if (cmd.empty()) {
        std::cerr << "Usage: " << argv[0] << " --cmd \"command to run\" [--warmup N] [--iters N] [--out file.csv]" << std::endl;
        return 1;
    }

    std::cout << "Starting benchmark: " << cmd << std::endl;
    std::cout << "Warmup: " << warmup << " iterations, Measurement: " << iters << " iterations" << std::endl;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        run_command(cmd);
    }

    std::vector<double> latencies;
    std::vector<double> memory_usages;
    latencies.reserve(iters);
    memory_usages.reserve(iters);

    // Measurement
    for (int i = 0; i < iters; ++i) {
        RunResult res = run_command(cmd);
        
        if (res.exit_code != 0) {
            std::cerr << "Warning: command returned non-zero exit code: " << res.exit_code << std::endl;
        }

        latencies.push_back(res.latency_ms);
        memory_usages.push_back(static_cast<double>(res.max_rss_kb));
        
        if ((i + 1) % 10 == 0) {
            std::cout << "Progress: " << (i + 1) << "/" << iters << "\r" << std::flush;
        }
    }
    std::cout << std::endl;

    Stats s = calculate_stats(latencies);
    Stats mem_s = calculate_stats(memory_usages);

    // Sample vitals once at the end
    std::string vitals_json = "";
    if (use_sysfs) {
        vitals_json = exec_vitals("python3 tools/read_sysfs.py");
    }

    // Output results
    std::ofstream ofs(out_file);
    ofs << "metric,value" << std::endl;
    ofs << "cmd,\"" << cmd << "\"" << std::endl;
    ofs << "iters," << iters << std::endl;
    ofs << "latency_min_ms," << s.min << std::endl;
    ofs << "latency_max_ms," << s.max << std::endl;
    ofs << "latency_mean_ms," << s.mean << std::endl;
    ofs << "latency_p50_ms," << s.p50 << std::endl;
    ofs << "latency_p90_ms," << s.p90 << std::endl;
    ofs << "latency_p99_ms," << s.p99 << std::endl;
    
    ofs << "memory_min_kb," << mem_s.min << std::endl;
    ofs << "memory_max_kb," << mem_s.max << std::endl;
    ofs << "memory_mean_kb," << mem_s.mean << std::endl;
    ofs << "memory_p50_kb," << mem_s.p50 << std::endl;
    
    if (!vitals_json.empty()) {
        // Very crude JSON parsing for key-value pairs
        // Expected: {"key": value, "key": value}
        std::string cleaned = vitals_json;
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '{'), cleaned.end());
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '}'), cleaned.end());
        cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\"'), cleaned.end());
        
        std::stringstream ss(cleaned);
        std::string item;
        while (std::getline(ss, item, ',')) {
            size_t colon = item.find(':');
            if (colon != std::string::npos) {
                std::string key = item.substr(0, colon);
                std::string val = item.substr(colon + 1);
                // trim spaces
                key.erase(0, key.find_first_not_of(" "));
                key.erase(key.find_last_not_of(" ") + 1);
                val.erase(0, val.find_first_not_of(" "));
                val.erase(val.find_last_not_of(" ") + 1);
                ofs << "vitals_" << key << "," << val << std::endl;
            }
        }
    }

    std::cout << "Results saved to " << out_file << std::endl;
    std::cout << "p50: " << s.p50 << " ms, p90: " << s.p90 << " ms, p99: " << s.p99 << " ms" << std::endl;

    return 0;
}
