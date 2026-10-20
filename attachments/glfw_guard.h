#pragma once

// Include this after the chapter's own GLFW include, so that GLFW_INCLUDE_VULKAN
// is already defined when GLFW is first included.
#include <GLFW/glfw3.h>

#include <stdexcept>

// Owns the GLFW lifetime: it initialises GLFW in the constructor and terminates
// it in the destructor.
//
// Declare it as the first member of the application class. Members are
// destroyed in reverse declaration order, so the first one declared is
// destroyed last - after every vk::raii member. The swapchain and surface still
// reference the window system connection when their destructors run, so
// glfwTerminate() has to outlive them. It also destroys any windows that are
// still open, so no separate glfwDestroyWindow() call is needed.
//
// If glfwInit() fails, the constructor throws and the destructor never runs, so
// glfwTerminate() is correctly skipped.
struct GlfwGuard
{
	GlfwGuard()
	{
		if (!glfwInit())
		{
			throw std::runtime_error("failed to initialize GLFW!");
		}
	}

	~GlfwGuard()
	{
		glfwTerminate();
	}
};
