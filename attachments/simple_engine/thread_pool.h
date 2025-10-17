#pragma once

#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <type_traits>
#include <tuple>
#include <utility>
#include <atomic>

// Generic reusable thread pool for background tasks (texture uploads, geometry processing, etc.)
class ThreadPool {
public:
    explicit ThreadPool(size_t threadCount = std::thread::hardware_concurrency())
        : stopFlag(false) {
        if (threadCount == 0) threadCount = 1;
        for (size_t i = 0; i < threadCount; ++i) {
            workers.emplace_back([this]() { this->workerLoop(); });
        }
    }

    ~ThreadPool() {
        shutdown();
    }

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            [func = std::decay_t<F>(std::forward<F>(f)),
             tup = std::make_tuple(std::forward<Args>(args)...)]() mutable -> return_type {
                return std::apply(std::move(func), std::move(tup));
            }
        );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stopFlag) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        condVar.notify_one();
        return res;
    }

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            if (stopFlag) return;
            stopFlag = true;
        }
        condVar.notify_all();
        for (auto& t : workers) {
            if (t.joinable()) t.join();
        }
        workers.clear();
    }

private:
    void workerLoop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                condVar.wait(lock, [this]() { return stopFlag || !tasks.empty(); });
                if (stopFlag && tasks.empty()) return;
                task = std::move(tasks.front());
                tasks.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condVar;
    std::atomic<bool> stopFlag;
};
