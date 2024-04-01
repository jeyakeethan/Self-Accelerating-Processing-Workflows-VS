class PdhCPUCounter {
    public:

        // @param counter_name: "\\Process(*ProcessName*)\\% Processor Time"
        explicit PdhCPUCounter(const std::string& counter_name);

        // Release handles here
        virtual ~PdhCPUCounter();

        // Provide actual CPU usage value in range [0.0, 1.0]
        double getCPUUtilization() const;

    private:

        // Low-level query
        PDH_FMT_COUNTERVALUE getFormattedCounterValue() const;

        // Needed for calculation
        size_t m_threads;

        // Counter format: "\\Process(*ProcessName*)\\% Processor Time"
        std::string m_counter_name;

        // CPU counter handle
        PDH_HCOUNTER m_counter = INVALID_HANDLE_VALUE;

        // Query to PerfMon handle
        PDH_HQUERY m_query = INVALID_HANDLE_VALUE;
    };
