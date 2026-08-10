#include <memory>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

class HelloTriangleApplication
{
  public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	// Owns the GLFW lifetime: initialises it here and terminates it in the
	// destructor. Declared first, so it is destroyed last - after every member
	// below. Later chapters add Vulkan objects here that still reference the
	// window system connection while they are destroyed, so glfwTerminate() has
	// to outlive them. It also destroys any windows that are still open.
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
	} glfwGuard;

	GLFWwindow *window = nullptr;

	void initWindow()
	{
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void initVulkan()
	{
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
		}
	}

	void cleanup()
	{
		// GLFW is torn down by glfwGuard, which is destroyed after every other
		// member. Terminating GLFW here would free the window system connection
		// while later chapters' Vulkan objects are still alive.
	}
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
