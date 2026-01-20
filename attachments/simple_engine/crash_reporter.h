/* Copyright (c) 2025 Holochip Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#	include <windows.h>
#	include <dbghelp.h>
#	pragma comment(lib, "dbghelp.lib")
#elif defined(__APPLE__) || defined(__linux__)
#	include <execinfo.h>
#	include <signal.h>
#	include <unistd.h>
#endif

#include "debug_system.h"

/**
 * @brief Class for crash reporting and minidump generation.
 *
 * This class implements the crash reporting system as described in the Tooling chapter:
 * @see en/Building_a_Simple_Engine/Tooling/04_crash_minidump.adoc
 */
class CrashReporter
{
  public:
	/**
	 * @brief Get the singleton instance of the crash reporter.
	 * @return Reference to the crash reporter instance.
	 */
	static CrashReporter &GetInstance()
	{
		static CrashReporter instance;
		return instance;
	}

	/**
	 * @brief Initialize the crash reporter.
	 * @param minidumpDir The directory to store minidumps.
	 * @param appName The name of the application.
	 * @param appVersion The version of the application.
	 * @return True if initialization was successful, false otherwise.
	 */
	bool Initialize(const std::string &minidumpDir = "crashes",
	                const std::string &appName     = "SimpleEngine",
	                const std::string &appVersion  = "1.0.0")
	{
		std::lock_guard<std::mutex> lock(mutex);

		this->minidumpDir = minidumpDir;
		this->appName     = appName;
		this->appVersion  = appVersion;

// Create minidump directory if it doesn't exist
#ifdef _WIN32
		CreateDirectoryA(minidumpDir.c_str(), NULL);
#else
		std::string command = "mkdir -p " + minidumpDir;
		system(command.c_str());
#endif

		// Install crash handlers
		InstallCrashHandlers();

		// Register with debug system
		DebugSystem::GetInstance().SetCrashHandler([this](const std::string &message) {
			this->HandleCrash(message);
		});

		LOG_INFO("CrashReporter", "Crash reporter initialized");
		initialized = true;
		return true;
	}

	/**
	 * @brief Clean up crash reporter resources.
	 */
	void Cleanup()
	{
		std::lock_guard<std::mutex> lock(mutex);

		if (initialized)
		{
			// Uninstall crash handlers
			UninstallCrashHandlers();

			LOG_INFO("CrashReporter", "Crash reporter shutting down");
			initialized = false;
		}
	}

	/**
	 * @brief Handle a crash.
	 * @param message The crash message.
	 */
	void HandleCrash(const std::string &message)
	{
		HandleCrashInternal(message, nullptr);
	}

	/**
	 * @brief Register a crash callback.
	 * @param callback The callback function to be called when a crash occurs.
	 * @return An ID that can be used to unregister the callback.
	 */
	int RegisterCrashCallback(std::function<void(const std::string &)> callback)
	{
		std::lock_guard<std::mutex> lock(mutex);

		int id             = nextCallbackId++;
		crashCallbacks[id] = callback;
		return id;
	}

	/**
	 * @brief Unregister a crash callback.
	 * @param id The ID of the callback to unregister.
	 */
	void UnregisterCrashCallback(int id)
	{
		std::lock_guard<std::mutex> lock(mutex);

		crashCallbacks.erase(id);
	}

	/**
	 * @brief Generate a minidump.
	 * @param message The crash message.
	 */
	void GenerateMinidump(const std::string &message, void *platformExceptionPointers = nullptr)
	{
		// Get current time for filename
		auto now  = std::chrono::system_clock::now();
		auto time = std::chrono::system_clock::to_time_t(now);
		char timeStr[20];
		std::strftime(timeStr, sizeof(timeStr), "%Y%m%d_%H%M%S", std::localtime(&time));

		// Create minidump filename
		std::string filename = minidumpDir + "/" + appName + "_" + timeStr + ".dmp";
		std::string report   = minidumpDir + "/" + appName + "_" + timeStr + ".txt";

		// Also write a small sidecar text file so users can quickly see the exception code/address
		// without needing a debugger.
		try
		{
			std::ofstream rep(report, std::ios::out | std::ios::trunc);
			rep << "Crash Report for " << appName << " " << appVersion << "\n";
			rep << "Timestamp: " << timeStr << "\n";
			rep << "Message: " << message << "\n";
#ifdef _WIN32
			if (platformExceptionPointers)
			{
				auto *exPtrs = reinterpret_cast<EXCEPTION_POINTERS *>(platformExceptionPointers);
				if (exPtrs && exPtrs->ExceptionRecord)
				{
					rep << "ExceptionCode: 0x" << std::hex << exPtrs->ExceptionRecord->ExceptionCode << std::dec << "\n";
					rep << "ExceptionAddress: " << exPtrs->ExceptionRecord->ExceptionAddress << "\n";
					rep << "ExceptionFlags: 0x" << std::hex << exPtrs->ExceptionRecord->ExceptionFlags << std::dec << "\n";
				}
			}
#endif
		}
		catch (...)
		{
		}

// Generate minidump based on platform
#ifdef _WIN32
		// Windows implementation
		EXCEPTION_POINTERS *exPtrs = reinterpret_cast<EXCEPTION_POINTERS *>(platformExceptionPointers);
		HANDLE hFile = CreateFileA(
		    filename.c_str(),
		    GENERIC_WRITE,
		    0,
		    NULL,
		    CREATE_ALWAYS,
		    FILE_ATTRIBUTE_NORMAL,
		    NULL);

		if (hFile != INVALID_HANDLE_VALUE)
		{
			MINIDUMP_EXCEPTION_INFORMATION exInfo{};
			exInfo.ThreadId          = GetCurrentThreadId();
			exInfo.ExceptionPointers = exPtrs;
			exInfo.ClientPointers    = FALSE;

			MINIDUMP_EXCEPTION_INFORMATION *exInfoPtr = exPtrs ? &exInfo : nullptr;
			MiniDumpWriteDump(GetCurrentProcess(),
			                  GetCurrentProcessId(),
			                  hFile,
			                  MiniDumpNormal,
			                  exInfoPtr,
			                  NULL,
			                  NULL);

			CloseHandle(hFile);
		}
#else
		// Unix implementation
		std::ofstream file(filename, std::ios::out | std::ios::binary);
		if (file.is_open())
		{
			// Get backtrace
			void  *callstack[128];
			int    frames  = backtrace(callstack, 128);
			char **symbols = backtrace_symbols(callstack, frames);

			// Write header
			file << "Crash Report for " << appName << " " << appVersion << std::endl;
			file << "Timestamp: " << timeStr << std::endl;
			file << "Message: " << message << std::endl;
			file << std::endl;

			// Write backtrace
			file << "Backtrace:" << std::endl;
			for (int i = 0; i < frames; i++)
			{
				file << symbols[i] << std::endl;
			}

			free(symbols);
			file.close();
		}
#endif

		// Best-effort stderr note (stdout/stderr redirection will capture this even if DebugSystem isn't initialized)
		std::fprintf(stderr, "[CrashReporter] Wrote minidump: %s\n", filename.c_str());
		std::fprintf(stderr, "[CrashReporter] Wrote report:  %s\n", report.c_str());
		std::fflush(stderr);
	}

  private:
	// Private constructor for singleton
	CrashReporter() = default;

	// Delete copy constructor and assignment operator
	CrashReporter(const CrashReporter &)            = delete;
	CrashReporter &operator=(const CrashReporter &) = delete;

	// Mutex for thread safety
	std::mutex mutex;

	// Initialization flag
	bool initialized = false;

	// Minidump directory
	std::string minidumpDir = "crashes";

	// Application info
	std::string appName    = "SimpleEngine";
	std::string appVersion = "1.0.0";

	// Crash callbacks
	std::unordered_map<int, std::function<void(const std::string &)>> crashCallbacks;
	int                                                               nextCallbackId = 0;
	std::atomic<bool>                                                 handlingCrash{false};

#ifdef _WIN32
	static bool ShouldCaptureException(EXCEPTION_POINTERS *exInfo, bool unhandled)
	{
		if (unhandled)
			return true;
		if (!exInfo || !exInfo->ExceptionRecord)
			return false;
		const DWORD code  = exInfo->ExceptionRecord->ExceptionCode;
		const DWORD flags = exInfo->ExceptionRecord->ExceptionFlags;
		// Ignore common first-chance C++ exceptions and breakpoint exceptions.
		if (code == 0xE06D7363u /* MSVC C++ EH */ || code == 0x80000003u /* breakpoint */)
			return false;
		// Capture likely-fatal errors and non-continuable exceptions.
		if ((flags & EXCEPTION_NONCONTINUABLE) != 0)
			return true;
		switch (code)
		{
			case 0xC0000409u: // STATUS_STACK_BUFFER_OVERRUN
			case 0xC0000005u: // STATUS_ACCESS_VIOLATION
			case 0xC000001Du: // STATUS_ILLEGAL_INSTRUCTION
			case 0xC00000FDu: // STATUS_STACK_OVERFLOW
			case 0xC0000374u: // STATUS_HEAP_CORRUPTION
				return true;
			default:
				return false;
		}
	}
#endif

#ifdef _WIN32
	void *vectoredHandlerHandle = nullptr;
#endif

	void HandleCrashInternal(const std::string &message, void *platformExceptionPointers)
	{
		bool expected = false;
		if (!handlingCrash.compare_exchange_strong(expected, true))
		{
			// Already handling a crash; avoid recursion.
			return;
		}
		std::lock_guard<std::mutex> lock(mutex);

		std::string msg = message;
		(void) platformExceptionPointers;

#ifdef _WIN32
		if (platformExceptionPointers)
		{
			auto *exPtrs = reinterpret_cast<EXCEPTION_POINTERS *>(platformExceptionPointers);
			if (exPtrs && exPtrs->ExceptionRecord)
			{
				const DWORD code = exPtrs->ExceptionRecord->ExceptionCode;
				void *addr       = exPtrs->ExceptionRecord->ExceptionAddress;
				char buf[128];
				std::snprintf(buf, sizeof(buf), " (code=0x%08lX, addr=%p)", static_cast<unsigned long>(code), addr);
				msg += buf;
			}
		}
#endif

		LOG_FATAL("CrashReporter", "Crash detected: " + msg);

		// Generate minidump
		GenerateMinidump(msg, platformExceptionPointers);

		// Call registered callbacks
		for (const auto &callback : crashCallbacks)
		{
			callback.second(msg);
		}
	}

	/**
	 * @brief Install platform-specific crash handlers.
	 */
	void InstallCrashHandlers()
	{
#ifdef _WIN32
		// Windows implementation
		// Vectored handler runs before SEH/unhandled filters and is more likely to fire for fast-fail style crashes.
		vectoredHandlerHandle = AddVectoredExceptionHandler(1, [](EXCEPTION_POINTERS *exInfo) -> LONG {
			if (CrashReporter::ShouldCaptureException(exInfo, /*unhandled=*/false))
			{
				CrashReporter::GetInstance().HandleCrashInternal("Vectored exception", exInfo);
			}
			return EXCEPTION_CONTINUE_SEARCH;
		});
		SetUnhandledExceptionFilter([](EXCEPTION_POINTERS *exInfo) -> LONG {
			CrashReporter::GetInstance().HandleCrashInternal("Unhandled exception", exInfo);
			return EXCEPTION_EXECUTE_HANDLER;
		});
#else
		// Unix implementation
		signal(SIGSEGV, [](int sig) {
			CrashReporter::GetInstance().HandleCrash("Segmentation fault");
			exit(1);
		});

		signal(SIGABRT, [](int sig) {
			CrashReporter::GetInstance().HandleCrash("Abort");
			exit(1);
		});

		signal(SIGFPE, [](int sig) {
			CrashReporter::GetInstance().HandleCrash("Floating point exception");
			exit(1);
		});

		signal(SIGILL, [](int sig) {
			CrashReporter::GetInstance().HandleCrash("Illegal instruction");
			exit(1);
		});
#endif
	}

	/**
	 * @brief Uninstall platform-specific crash handlers.
	 */
	void UninstallCrashHandlers()
	{
#ifdef _WIN32
		// Windows implementation
		SetUnhandledExceptionFilter(NULL);
		if (vectoredHandlerHandle)
		{
			RemoveVectoredExceptionHandler(vectoredHandlerHandle);
			vectoredHandlerHandle = nullptr;
		}
#else
		// Unix implementation
		signal(SIGSEGV, SIG_DFL);
		signal(SIGABRT, SIG_DFL);
		signal(SIGFPE, SIG_DFL);
		signal(SIGILL, SIG_DFL);
#endif
	}
};

// Convenience macro for simulating a crash (for testing)
#define SIMULATE_CRASH(message) CrashReporter::GetInstance().HandleCrash(message)
