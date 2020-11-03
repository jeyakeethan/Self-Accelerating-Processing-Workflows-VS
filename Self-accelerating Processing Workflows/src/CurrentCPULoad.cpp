#include <Windows.h>
#include <iostream>
#include <thread>
#include <sstream>
#include <Constants.h>
using namespace std;
static float CalculateCPULoad(unsigned long long idleTicks, unsigned long long totalTicks);
static unsigned long long FileTimeToInt64(const FILETIME& ft);

class CurrentCPULoad {
    public:
        FILETIME idleTime, kernelTime, userTime;
        unsigned long long last_idle_ticks;
        float load;
        thread updater;
        CurrentCPULoad() { 
        }
        ~CurrentCPULoad() { updater.join();
        }
        float GetCPULoad()
        {
            load = GetSystemTimes(&idleTime, &kernelTime, &userTime)? CalculateCPULoad(FileTimeToInt64(idleTime), FileTimeToInt64(kernelTime) + FileTimeToInt64(userTime)) : -1.0f;
            return load;
        }

        void UpdateIdleTime()
        {
            GetSystemTimes(&idleTime, NULL, NULL);
        }
        static void x(CurrentCPULoad y) {
            while (true)
            {
                auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(CPU_LOAD_REVISE_PERIOD);
                y.UpdateIdleTime();
                std::this_thread::sleep_until(x);
            }
        }
        long long GetIdleTime()
        {
            return FileTimeToInt64(idleTime);
        }
        static string GetTimeStamp()
        {
            SYSTEMTIME sysTime;
            GetSystemTime(&sysTime);
            stringstream s;
            s << sysTime.wHour << ":" << sysTime.wMinute << ":" << sysTime.wSecond;
            return s.str();
        }
        static int GetTimeHour()
        {
            SYSTEMTIME sysTime;
            GetSystemTime(&sysTime);
            return sysTime.wHour;
        }

        float CalculateCPULoad(unsigned long long idleTicks, unsigned long long totalTicks)
        {
            static unsigned long long _previousTotalTicks = 0;
            static unsigned long long _previousIdleTicks = 0;

            unsigned long long totalTicksSinceLastTime = totalTicks - _previousTotalTicks;
            unsigned long long idleTicksSinceLastTime = idleTicks - _previousIdleTicks;


            float ret = 1.0f - ((totalTicksSinceLastTime > 0) ? ((float)idleTicksSinceLastTime) / totalTicksSinceLastTime : 0);

            _previousTotalTicks = totalTicks;
            _previousIdleTicks = idleTicks;
            return ret;
        }

        static unsigned long long FileTimeToInt64(const FILETIME& ft)
        {
            return (((unsigned long long)(ft.dwHighDateTime)) << 32) | ((unsigned long long)ft.dwLowDateTime);
        }

    private:
};

static float CalculateCPULoad(unsigned long long idleTicks, unsigned long long totalTicks)
{
    static unsigned long long _previousTotalTicks = 0;
    static unsigned long long _previousIdleTicks = 0;

    unsigned long long totalTicksSinceLastTime = totalTicks - _previousTotalTicks;
    unsigned long long idleTicksSinceLastTime = idleTicks - _previousIdleTicks;


    float ret = 1.0f - ((totalTicksSinceLastTime > 0) ? ((float)idleTicksSinceLastTime) / totalTicksSinceLastTime : 0);

    _previousTotalTicks = totalTicks;
    _previousIdleTicks = idleTicks;
    return ret;
}

static unsigned long long FileTimeToInt64(const FILETIME& ft)
{
    return (((unsigned long long)(ft.dwHighDateTime)) << 32) | ((unsigned long long)ft.dwLowDateTime);
}