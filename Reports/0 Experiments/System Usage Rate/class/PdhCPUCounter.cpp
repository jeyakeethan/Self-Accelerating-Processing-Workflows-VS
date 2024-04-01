PdhCPUCounter::PdhCPUCounter(const std::string& counter_name) :
        m_counter_name(counter_name),
        m_threads(std::thread::hardware_concurrency())
    {
        PdhOpenQuery(nullptr, 0, &m_query);
        PdhAddEnglishCounter(m_query, m_counter_name.c_str(), 0, &m_counter);
        PdhCollectQueryData(m_query);
    }

    PDH_FMT_COUNTERVALUE PdhCPUCounter::getFormattedCounterValue() const
    {
        PdhCollectQueryData(m_query);

        PDH_FMT_COUNTERVALUE val;
        PdhGetFormattedCounterValue(m_counter, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100, nullptr, &val);
        return val;
    }

    double PdhCPUCounter::getCPUUtilization() const
    {
        const auto &val = getFormattedCounterValue();
        return val.doubleValue / m_threads;
    }
