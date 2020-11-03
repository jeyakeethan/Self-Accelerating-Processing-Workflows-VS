#include <Windows.h>
#include <iostream>
#include <thread>
#include <sstream>
#include <Constants.h>
using namespace std;

static float CalculateCPULoad();
static unsigned long long FileTimeToInt64();
float GetCPULoad();

class CurrentCPULoad {
public:
    static FILETIME idleTime, kernelTime, userTime;
    static unsigned long long last_idle_ticks;
    static float load;
    CurrentCPULoad(): done(false) {}
    ~CurrentCPULoad() {
        done = true; thr.join();
    }
    static float GetCPULoad()
    {
        CurrentCPULoad::load = GetSystemTimes(&CurrentCPULoad::idleTime, &CurrentCPULoad::kernelTime, &CurrentCPULoad::userTime)?CalculateCPULoad(FileTimeToInt64(CurrentCPULoad::idleTime), FileTimeToInt64(CurrentCPULoad::kernelTime) + FileTimeToInt64(CurrentCPULoad::userTime)) : -1.0f;
    }
    static float UpdateIdleTime()
    {
        GetSystemTimes(&idleTime, NULL, NULL);
    }
    static long long GetIdleTime()
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
    static void startCPULoadUpdateThread() {
        CurrentCPULoad::thr = thread{ [&]() {
                while (true)
                {
                    auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(CPU_LOAD_REVISE_PERIOD);
                    CurrentCPULoad::UpdateIdleTime();
                    std::this_thread::sleep_until(x);
                }
            } };
        CurrentCPULoad::thr.detach();
    }
private:
    static thread thr;
    bool done;
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

// Returns 1.0f for "CPU fully pinned", 0.0f for "CPU idle", or somewhere in between
// You'll need to call this at regular intervals, since it measures the load between
// the previous call and the current one.  Returns -1.0 on error.
