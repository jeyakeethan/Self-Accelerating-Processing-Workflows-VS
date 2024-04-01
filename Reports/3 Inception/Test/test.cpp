#include <iostream>
#include <chrono>
#include <thread>
#include <functional>


void timer_start(std::function<void(void)> func, unsigned int interval)
{
  std::thread([func, interval]()
  {
    while (true)
    {
      auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval);
      func();
      std::this_thread::sleep_until(x);
    }
  }).detach();
}

void do_something()
{
  std::cout << "I am doing something" << std::endl;
}

int main()
{
  timer_start(do_something, 1000);
  while (true);
}
