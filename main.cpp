#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdint>  // for UINT32_MAX
#include <fstream>

#include <cstdlib>
using std::cout;
using std::endl;

#include <algorithm>
using std::min;
using std::max;

#include <cstring>
using std::string;

#include <vector>
using std::vector;

#include <set>
using std::set;

#include <optional>
using std::optional;

const int SCALE = 2;
const int WIDTH  = 800*SCALE;
const int HEIGHT = 600*SCALE;

// need char** (array of char*) for VkInstanceCreateInfo
const vector<const char*> validation_layer_names = {
		"VK_LAYER_KHRONOS_validation"
};
const vector<const char*> device_extension_names = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

const bool verbose_init = true;

#ifdef NDEBUG
	const bool enable_validation_layers = false;
#else
	const bool enable_validation_layers = true;
#endif

/// Helper Proxy functions for access to Extension functions
/// These could be static class methods, or outside of class - I think.

/// Link my static debugCallback into Vulkan using VK_EXT_debug_utils extension
///   create_info struct includes pointer to my static debugCallback
///   p_allocator is null
///   p_debug_messenger is returned, and used by destroyDebugUtilsMessenger
VkResult createDebugUtilsMessengerEXT(
		VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
		const VkAllocationCallbacks*              p_allocator,
		VkDebugUtilsMessengerEXT*                 p_debug_messenger) {

	// get a pointer to the desired function exposed by the VK_EXT_debug_utils extension
	auto pfn_create_debug_messenger =
			(PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
			instance,
			"vkCreateDebugUtilsMessengerEXT");

	// call the extension function to link in my debugCallback
	if (pfn_create_debug_messenger != nullptr)
	{
		return pfn_create_debug_messenger(
				instance, p_create_info, p_allocator, p_debug_messenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

/// Unlink my static debugCallback using the VK_EXT_debug_utils extension
void destroyDebugUtilsMessengerEXT(
		VkInstance instance,
		VkDebugUtilsMessengerEXT debug_messenger, // not to be confused with class member
		const VkAllocationCallbacks* p_allocator) {

	// get a pointer to the desired function exposed by the VK_EXT_debug_utils extension
	auto pfn_destroy_debug_messenger = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
			instance,
			"vkDestroyDebugUtilsMessengerEXT");

	// call the extension function to unlink in my debugCallback
	if (pfn_destroy_debug_messenger != nullptr) {
		pfn_destroy_debug_messenger(instance, debug_messenger, p_allocator);
	} else {
		std::cerr << "Failed to get VK_EXT_debug_utils pfn to destroy debug messenger" << endl;
	}
}

struct QueueFamilyIndices {
	optional<uint32_t> graphics_family;
	optional<uint32_t> present_family;

	bool isComplete() {
		return graphics_family.has_value() && present_family.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR   capabilities;
	vector<VkSurfaceFormatKHR> formats;
	vector<VkPresentModeKHR>   present_modes;
};

class HelloTriangleApplication {
public:   // members

private:  // members
	GLFWwindow*  m_window;
	VkInstance   m_instance;
	VkSurfaceKHR m_surface;

	VkDebugUtilsMessengerEXT m_debug_messenger;

	VkPhysicalDevice m_physical_device;
	VkDevice         m_logical_device;

	VkQueue  m_graphics_queue;
	VkQueue  m_present_queue;

	VkSwapchainKHR  m_swap_chain;
	vector<VkImage> m_swap_chain_images; // auto destroyed - no clean up required
	VkFormat        m_swap_chain_image_format;
	VkExtent2D      m_swap_chain_extent;
	vector<VkImageView> m_swap_chain_image_views;

	VkRenderPass     m_render_pass;
	VkPipelineLayout m_pipeline_layout;
	VkPipeline       m_graphics_pipeline;

	vector<VkFramebuffer> m_swap_chain_framebuffers;

	VkCommandPool m_command_pool;
	vector<VkCommandBuffer> m_command_buffers;

public:   // Methods

	/// constructor -
	HelloTriangleApplication() {
		m_window   = nullptr;
		m_instance = nullptr;
		m_surface  = VK_NULL_HANDLE;

		m_debug_messenger = nullptr;

		m_physical_device = VK_NULL_HANDLE;
		m_logical_device  = VK_NULL_HANDLE;

		m_graphics_queue  = VK_NULL_HANDLE;
		m_present_queue   = VK_NULL_HANDLE;

		m_swap_chain      = VK_NULL_HANDLE;
		m_swap_chain_image_format = VK_FORMAT_UNDEFINED;
		m_swap_chain_extent = {0,0};

		m_render_pass       = VK_NULL_HANDLE;
		m_pipeline_layout   = VK_NULL_HANDLE;
		m_graphics_pipeline = VK_NULL_HANDLE;

		m_command_pool = VK_NULL_HANDLE;
	}


	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}


private:    // methods
	void initWindow(){
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		m_window = glfwCreateWindow(
				WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createCommandBuffers();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(m_window)){
			glfwPollEvents();
		}
	}

	void cleanup(){

		vkDestroyCommandPool(m_logical_device, m_command_pool, nullptr);

		for (auto framebuffer :  m_swap_chain_framebuffers) {
			vkDestroyFramebuffer(m_logical_device, framebuffer, nullptr);
		}

		vkDestroyPipeline(m_logical_device, m_graphics_pipeline, nullptr);
		vkDestroyPipelineLayout(m_logical_device, m_pipeline_layout, nullptr);
		vkDestroyRenderPass(m_logical_device, m_render_pass, nullptr);

		for (auto image_view : m_swap_chain_image_views) {
			vkDestroyImageView(m_logical_device, image_view, nullptr);
		}
		vkDestroySwapchainKHR(m_logical_device, m_swap_chain, nullptr);
		vkDestroyDevice(m_logical_device, nullptr);

		if (enable_validation_layers) {
			destroyDebugUtilsMessengerEXT(m_instance, m_debug_messenger, nullptr);
		}

		vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
		vkDestroyInstance(m_instance, nullptr);

		glfwDestroyWindow(m_window);
		glfwTerminate();
	}

	void createInstance() {
		string validation_error;
		if (enable_validation_layers &&
		    !checkValidationLayerSupport(validation_error)) {
			throw std::runtime_error(
					validation_error + "check ValidationLayerSupport() failed");
		}

		VkApplicationInfo app_info = {};
		app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.pApplicationName   = "Hello Triangle";
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pEngineName        = "No Engine";
		app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
		app_info.apiVersion         = VK_API_VERSION_1_0;

		VkInstanceCreateInfo instance_create_info = {};
		instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instance_create_info.pApplicationInfo = &app_info;

		auto extensions = getRequiredExtensions();
		uint32_t extension_count = extensions.size();
		instance_create_info.enabledExtensionCount = extension_count;
		instance_create_info.ppEnabledExtensionNames = extensions.data();

		// Enable debugCallback prior to instance creation
		VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {};

		if (enable_validation_layers) {
			instance_create_info.enabledLayerCount =
					static_cast<uint32_t>(validation_layer_names.size());
			instance_create_info.ppEnabledLayerNames = validation_layer_names.data();

			// Create a debugMessenger for use during vkCreateInstance and vkDestroyInstance
			populateDebugMessengerCreateInfo(debug_create_info);
			instance_create_info.pNext =
					(VkDebugUtilsMessengerCreateInfoEXT*) &debug_create_info;

		} else {
			instance_create_info.enabledLayerCount = 0;
			instance_create_info.pNext = nullptr;
		}

		if (verbose_init) {
			listValidationLayerNames(validation_layer_names);
			listAvailableVkExtensions();
			listRequiredVkExtensions(extension_count, extensions.data());
		}

		if (vkCreateInstance(&instance_create_info, nullptr, &m_instance) != VK_SUCCESS) {
			throw std::runtime_error("createInstance() failed to create m_instance!");
		}

		if (verbose_init) {
			cout << "createInstance() VK_SUCCESS" << endl;
		}
	}

	static void listValidationLayerNames(vector<const char*> validation_layer_names) {
		cout << "Enabled Validation Layers: " << validation_layer_names.size() << endl;
		for (const auto& layer_name : validation_layer_names){
			cout << "\t" << layer_name << endl;
		}
	}

	static void listRequiredVkExtensions(uint32_t extension_count, const char** extension_names) {
		cout << "Required Vk Extensions: " << extension_count << endl;
		for (auto i = 0; i < extension_count; i++) {
			cout << "\t" << extension_names[i] << endl;
		}
	}

	static void listAvailableVkExtensions(){
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(
				nullptr,
				&extensionCount,
				nullptr);
		vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(
				nullptr,
				&extensionCount,
				extensions.data());
		cout << "Available Vk Extensions: " << extensionCount << endl;
		for (const auto& extension : extensions) {
			cout << "\t" << extension.extensionName << endl;
		}
	}

	static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& create_info) {

		// create_info struct defines when to call my CallBack, and where my callback is.
		create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		create_info.messageSeverity =
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType =
				VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
				VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = debugCallback;
		create_info.pUserData = nullptr;
	}

	/// Use VK_EXT_debug_utils extension, to link in my own static debugCallback method
	void setupDebugMessenger() {
		if (!enable_validation_layers) return;

		// create_info struct defines when to call my CallBack, and where my callback is.
		VkDebugUtilsMessengerCreateInfoEXT  create_info = {};
		populateDebugMessengerCreateInfo(create_info);

		if (createDebugUtilsMessengerEXT(
				m_instance,
				&create_info,
				nullptr,
				&m_debug_messenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to setup debug messenger!");
		}
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
			VkDebugUtilsMessageSeverityFlagBitsEXT      message_severity,
			VkDebugUtilsMessageTypeFlagsEXT             message_type,
			const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
			void* p_user_data) {

		std::cerr << "debugCallback validation layer: " << endl
		          << "\t" << p_callback_data->pMessage << endl;

		// indicate whether Vulkan call triggering this callback should abort.
		return VK_FALSE;
	}

	void createSurface(){
		if (glfwCreateWindowSurface(
				m_instance,
				m_window,
				nullptr,
				&m_surface) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create window surface!");
		}
		if (verbose_init) {
			cout << "=== Surface Created ===" << endl;
		}
	}

	void pickPhysicalDevice() {
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(m_instance, &device_count, nullptr);
		if (device_count == 0) {
			throw std::runtime_error("failed to find GPU with Vulkan support!");
		}

		vector<VkPhysicalDevice> physical_devices(device_count);
		vkEnumeratePhysicalDevices(m_instance, &device_count, physical_devices.data());

		// enumerate devices
		if(verbose_init) {
			cout << endl << "Num Physical Devices: " << device_count << endl;
			for (int i = 0; i < device_count; i++) {
				VkPhysicalDeviceProperties device_props = {};
				vkGetPhysicalDeviceProperties(physical_devices[i], &device_props);
				bool device_suitable = isDeviceSuitable(physical_devices[i]);
				cout << "device[" << i << "] " << device_props.deviceName;
				cout << ": deviceIsSuitable() = ";
				cout << std::boolalpha << device_suitable << endl;
				printQueueFamilies(physical_devices[i]);
				cout << endl;
			}
		}

		// Select the first valid device.
		int selected_device = 0;
		for (const auto& device_to_test : physical_devices) {
			if (isDeviceSuitable(device_to_test)) {
				m_physical_device = device_to_test;
				break;
			}
			selected_device++;
		}

		if (m_physical_device == VK_NULL_HANDLE) {
			throw std::runtime_error("Failed to find suitable GPU!");
		}

		// radv exposes 2 devices, amdvlk exposes 1
		// override and select second radv device
		// the second radv device, behaves more like the amdvlk device
		if (device_count > 1) {
			selected_device = 1;  // selection override
			m_physical_device = physical_devices[selected_device];
		}

		if (verbose_init) {
			cout << "Selected device: " << selected_device << endl;
			QueueFamilyIndices indices = findQueueFamilies(m_physical_device);
			if (indices.graphics_family.has_value()) {
				cout << "Selected Queue Family :" << indices.graphics_family.value() << endl;
			} else {
				std::cerr << "Queue Family not selected!" << endl;
			}
		}
	}

	void createLogicalDevice(){
		QueueFamilyIndices indices = findQueueFamilies(m_physical_device);

		vector<VkDeviceQueueCreateInfo> queue_create_infos;
		set<uint32_t> unique_queue_families = {
				indices.graphics_family.value(),
				indices.present_family.value()
		};

		float queue_priority = 1.0f;
		for (uint32_t queue_family: unique_queue_families) {
			VkDeviceQueueCreateInfo queue_create_info = {};
			queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queue_create_info.queueFamilyIndex = queue_family;
			queue_create_info.queueCount = 1;
			queue_create_info.pQueuePriorities = &queue_priority;
			queue_create_infos.push_back(queue_create_info);
		}

		VkPhysicalDeviceFeatures device_features = {};

		VkDeviceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		create_info.queueCreateInfoCount =
				static_cast<uint32_t> (queue_create_infos.size());
		create_info.pQueueCreateInfos = queue_create_infos.data();
		create_info.pEnabledFeatures = &device_features;
		create_info.enabledExtensionCount =
				static_cast<uint32_t>(device_extension_names.size());
		create_info.ppEnabledExtensionNames = device_extension_names.data();

		// These are now ignored, but set for compatibility with older implementations.
		if (enable_validation_layers) {
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layer_names.size());
			create_info.ppEnabledLayerNames = validation_layer_names.data();
		} else {
			create_info.enabledLayerCount = 0;
		}

		// Create the logical device
		if (vkCreateDevice(m_physical_device, &create_info, nullptr, &m_logical_device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device");
		}

		vkGetDeviceQueue(
				m_logical_device, indices.graphics_family.value(),0, &m_graphics_queue);

		vkGetDeviceQueue(
				m_logical_device, indices.present_family.value(),0,	&m_present_queue);

		if (verbose_init) {
			cout << "=== Logical Device Created using: ===" << endl;
			cout << "  - Graphics Queue " << indices.graphics_family.value() << endl;
			cout << "  - Present Queue  " << indices.present_family.value() << endl;
		}
	}

	void createSwapChain() {
		SwapChainSupportDetails swap_chain_support =
				querySwapChainSupport(m_physical_device);
		VkSurfaceFormatKHR surface_format =
				chooseSwapSurfaceFormat(swap_chain_support.formats);
		VkPresentModeKHR present_mode =
				chooseSwapPresentMode(swap_chain_support.present_modes);
		VkExtent2D extent = chooseSwapExtent(swap_chain_support.capabilities);

		// add one for good measure - to avoid waiting on driver
		uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;

		// clamp to max (0 indicates no max)
		if (swap_chain_support.capabilities.maxImageCount > 0) {
			image_count = min(image_count, swap_chain_support.capabilities.maxImageCount);
		}

		VkSwapchainCreateInfoKHR create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		create_info.surface = m_surface;
		create_info.minImageCount    = image_count;
		create_info.imageFormat      = surface_format.format;
		create_info.imageColorSpace  = surface_format.colorSpace;
		create_info.imageExtent      = extent;
		create_info.imageArrayLayers = 1;
		create_info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(m_physical_device);
		uint32_t queue_family_indices[] {
				indices.graphics_family.value(),
				indices.present_family.value()
		};

		if (indices.graphics_family != indices.present_family) {
			create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
			create_info.queueFamilyIndexCount = 2;
			create_info.pQueueFamilyIndices   = queue_family_indices;
		} else {
			create_info.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
			create_info.queueFamilyIndexCount = 0;
			create_info.pQueueFamilyIndices   = nullptr;
		}

		create_info.preTransform   = swap_chain_support.capabilities.currentTransform;
		create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		create_info.presentMode    = present_mode;
		create_info.clipped        = VK_TRUE;
		create_info.oldSwapchain   = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(
				m_logical_device,
				&create_info,
				nullptr,
				&m_swap_chain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		// retrieve and save the handles to the swap chain images
		uint32_t actual_image_count = 0;
		vkGetSwapchainImagesKHR(
				m_logical_device,
				m_swap_chain,
				&actual_image_count,
				nullptr);
		m_swap_chain_images.resize(actual_image_count);
		vkGetSwapchainImagesKHR(
				m_logical_device,
				m_swap_chain,
				&actual_image_count,
				m_swap_chain_images.data());
		m_swap_chain_image_format = surface_format.format;
		m_swap_chain_extent = extent;

		if (verbose_init){
			cout << "=== Swap chain created ===" << endl;
			cout << "  - min_image_count    = " << image_count << endl;
			cout << "  - actual_image_count = " << actual_image_count << endl;
		}
	}

	void createImageViews(){

		size_t num_swap_chain_images = m_swap_chain_images.size();

		// size the views to match the images
		m_swap_chain_image_views.resize(num_swap_chain_images);

		VkImageViewCreateInfo create_info = {};
		create_info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		create_info.format   = m_swap_chain_image_format;
		create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		create_info.subresourceRange.aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT;
		create_info.subresourceRange.baseMipLevel = 0;
		create_info.subresourceRange.levelCount   = 1;
		create_info.subresourceRange.baseArrayLayer = 0;
		create_info.subresourceRange.layerCount = 1;

		for (size_t i = 0; i < num_swap_chain_images; i++) {
			create_info.image = m_swap_chain_images[i];

			if (vkCreateImageView(m_logical_device, &create_info, nullptr,
					&m_swap_chain_image_views[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}
		}

		if (verbose_init) {
			cout << "=== " << num_swap_chain_images << " Image Views Created ===" << endl;
		}
	}

	void createRenderPass() {
		VkAttachmentDescription color_attachment = {};
		color_attachment.format  = m_swap_chain_image_format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment_ref = {};
		color_attachment_ref.attachment = 0;  // index into attachment description array
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;

		VkRenderPassCreateInfo render_pass_info = {};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 1;
		render_pass_info.pAttachments = &color_attachment;
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses   = &subpass;

		if (vkCreateRenderPass(
				m_logical_device, &render_pass_info, nullptr, &m_render_pass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createGraphicsPipeline() {
		// local vector buffers freed at end of function scope
		auto vert_shader_code = readFile("../shaders/vert.spv");
		auto frag_shader_code = readFile("../shaders/frag.spv");

		// local module handles, so vkDestroyShaderModule required by end of function
		VkShaderModule vert_shader_module = createShaderModule(vert_shader_code);
		VkShaderModule frag_shader_module = createShaderModule(frag_shader_code);

		/// Shader Stage Creation
		VkPipelineShaderStageCreateInfo vert_shader_stage_info = {};
		vert_shader_stage_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vert_shader_stage_info.stage  = VK_SHADER_STAGE_VERTEX_BIT;
		vert_shader_stage_info.module = vert_shader_module;
		vert_shader_stage_info.pName  = "main"; // get to choose entry point
		vert_shader_stage_info.pSpecializationInfo = nullptr; // optional for shader constants

		VkPipelineShaderStageCreateInfo frag_shader_stage_info = {};
		frag_shader_stage_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		frag_shader_stage_info.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		frag_shader_stage_info.module = frag_shader_module;
		frag_shader_stage_info.pName  = "main"; // get to choose entry point
		frag_shader_stage_info.pSpecializationInfo = nullptr; // optional for shader constants

		VkPipelineShaderStageCreateInfo shader_stages[] = {
				vert_shader_stage_info, frag_shader_stage_info};

		/// Define vertex formats
		VkPipelineVertexInputStateCreateInfo vertex_input_info = {};
		vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_info.vertexBindingDescriptionCount = 0;
		vertex_input_info.pVertexBindingDescriptions = nullptr;
		vertex_input_info.vertexAttributeDescriptionCount = 0;
		vertex_input_info.pVertexAttributeDescriptions = nullptr;

		/// Define the primitive operation
		VkPipelineInputAssemblyStateCreateInfo input_assembly = {};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly.primitiveRestartEnable = VK_FALSE;

		/// Viewport and scissor -- one or more of each if supported by logical device
		VkViewport view_port = {};
		view_port.x = 0.0f;
		view_port.y = 0.0f;
		view_port.width = (float) m_swap_chain_extent.width;  // extent elements are int32
		view_port.height = (float) m_swap_chain_extent.height;
		view_port.minDepth = 0.0f;
		view_port.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = {0, 0};
		scissor.extent = m_swap_chain_extent;

		VkPipelineViewportStateCreateInfo view_port_state = {};
		view_port_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		view_port_state.viewportCount = 1;
		view_port_state.pViewports = &view_port;
		view_port_state.scissorCount = 1;
		view_port_state.pScissors = &scissor;

		/// Rasterizer - this is where you select wireframe
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // VK_POLYGON_MODE_LINE;
		rasterizer.lineWidth = 1.0f; // used by LINE or POINT modes. >1 requires wideLines GPU feature.
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		/// Multisampling - one approach to anti-aliasing - not used now
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		/// Depth and Stencil testing - not used now
		// VkPipelineDepthStencilStateCreateInfo depth_stencil = {};

		/// Color Blending
		VkPipelineColorBlendAttachmentState color_blend_attachment = {};
		color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_A_BIT |
				VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
		color_blend_attachment.blendEnable = VK_FALSE;
		color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // optional
		color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
		color_blend_attachment.colorBlendOp        = VK_BLEND_OP_ADD;      // optional
		color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // optional
		color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
		color_blend_attachment.alphaBlendOp        = VK_BLEND_OP_ADD;      // optional

		VkPipelineColorBlendStateCreateInfo color_blending = {};
		color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blending.logicOpEnable = VK_FALSE;
		color_blending.logicOp = VK_LOGIC_OP_COPY; // optional
		color_blending.attachmentCount = 1;
		color_blending.pAttachments = &color_blend_attachment;
		color_blending.blendConstants[0] = 0.0f; // optional
		color_blending.blendConstants[1] = 0.0f; // optional
		color_blending.blendConstants[2] = 0.0f; // optional
		color_blending.blendConstants[3] = 0.0f; // optional

		/// Dynamic State
		VkDynamicState dynamic_states[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_LINE_WIDTH
		};

		VkPipelineDynamicStateCreateInfo dynamic_state = {};
		dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_state.dynamicStateCount = 2;
		dynamic_state.pDynamicStates = dynamic_states;

		/// Pipeline layout
		VkPipelineLayoutCreateInfo pipeline_layout_info = {};
		pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_info.setLayoutCount = 0;       // optional
		pipeline_layout_info.pSetLayouts    = nullptr; // optional
		pipeline_layout_info.pushConstantRangeCount = 0;       // optional
		pipeline_layout_info.pPushConstantRanges    = nullptr; // optional

		if (vkCreatePipelineLayout(m_logical_device, &pipeline_layout_info,nullptr,
				&m_pipeline_layout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipeline_info = {};
		pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_info.stageCount = 2;
		pipeline_info.pStages = shader_stages;
		pipeline_info.pVertexInputState = &vertex_input_info;
		pipeline_info.pInputAssemblyState = &input_assembly;
		pipeline_info.pViewportState = &view_port_state;
		pipeline_info.pRasterizationState = &rasterizer;
		pipeline_info.pMultisampleState = &multisampling;
		pipeline_info.pDepthStencilState = nullptr;
		pipeline_info.pColorBlendState = &color_blending;
		pipeline_info.pDynamicState = nullptr;
		pipeline_info.layout = m_pipeline_layout;
		pipeline_info.renderPass = m_render_pass;
		pipeline_info.subpass = 0;
		pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
		pipeline_info.basePipelineIndex  = -1;

		if (vkCreateGraphicsPipelines(m_logical_device, VK_NULL_HANDLE, 1,
				&pipeline_info, nullptr, &m_graphics_pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		// VkShaderModule no longer required after pipeline created
		vkDestroyShaderModule(m_logical_device, frag_shader_module, nullptr);
		vkDestroyShaderModule(m_logical_device, vert_shader_module, nullptr);
	}

	void createFramebuffers(){
		// one framebuffer for each image_view
		m_swap_chain_framebuffers.resize(m_swap_chain_image_views.size());

		for (size_t i = 0; i < m_swap_chain_image_views.size(); i++) {
			VkImageView attachments[] = {m_swap_chain_image_views[i]}; // array of one?

			VkFramebufferCreateInfo frame_buffer_info = {};
			frame_buffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frame_buffer_info.renderPass = m_render_pass;
			frame_buffer_info.attachmentCount = 1;
			frame_buffer_info.pAttachments = attachments;
			frame_buffer_info.width  = m_swap_chain_extent.width;
			frame_buffer_info.height = m_swap_chain_extent.height;
			frame_buffer_info.layers = 1;

			if (vkCreateFramebuffer(m_logical_device, &frame_buffer_info, nullptr,
					&m_swap_chain_framebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}

		}
	}

	void createCommandPool(){
		QueueFamilyIndices queue_family_indices = findQueueFamilies(m_physical_device);

		VkCommandPoolCreateInfo pool_info = {};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();
		pool_info.flags = 0;

		if (vkCreateCommandPool(m_logical_device, &pool_info,nullptr,
				&m_command_pool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void createCommandBuffers(){
		// One command buffer for each framebuffer (or for each image view)
		m_command_buffers.resize(m_swap_chain_framebuffers.size());

		// Allocate enough empty command buffers -- and put them in pool
		VkCommandBufferAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.commandPool = m_command_pool;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandBufferCount = (uint32_t) m_command_buffers.size();

		if (vkAllocateCommandBuffers(m_logical_device, &alloc_info,
				m_command_buffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	static vector<char> readFile(const string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error(filename + " - failed to open file! ");
		}

		size_t file_size = (size_t) file.tellg(); // ios::ate opened at end of file
		vector<char> buffer(file_size);
		file.seekg(0);
		file.read(buffer.data(), file_size);
		file.close();

		if (verbose_init) {
			cout << filename << " loaded: " << file_size << " bytes" << endl;
		}

		return buffer;
	}

	VkShaderModule createShaderModule(const vector<char>& code) {
		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shader_module;
		if (vkCreateShaderModule(
				m_logical_device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}
		return shader_module;
	}

	static VkSurfaceFormatKHR chooseSwapSurfaceFormat(
			const vector<VkSurfaceFormatKHR>& available_formats) {

		// Look for desired format
		for (const auto& available_format : available_formats) {
			if (available_format.format     == VK_FORMAT_B8G8R8A8_SRGB &&
			    available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return available_format;
			}
		}

		// If desired not found, settle for first format specified
		return available_formats[0];
	}

	static VkPresentModeKHR chooseSwapPresentMode(
			const vector<VkPresentModeKHR>& available_present_modes) {

		// Try for Triple Buffering - least latency, best trade off
		for (const auto& available_present_mode : available_present_modes) {
			if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return available_present_mode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;  // guaranteed to be available
	}

	static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX) {
			return capabilities.currentExtent;
		} else {
			VkExtent2D actual_extent = {WIDTH, HEIGHT};
			actual_extent.width = max(
					capabilities.minImageExtent.width,
					min(capabilities.maxImageExtent.width, actual_extent.width));
			actual_extent.height = max(
					capabilities.minImageExtent.height,
					min(capabilities.maxImageExtent.height, actual_extent.height));
			return actual_extent;
		}
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device_to_query) {
		SwapChainSupportDetails details;

		// m_surface was created just after CreateInstance
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
				device_to_query, m_surface, &details.capabilities);

		uint32_t format_count;
		vkGetPhysicalDeviceSurfaceFormatsKHR(
				device_to_query, m_surface, &format_count, nullptr);
		if (format_count != 0) {
			details.formats.resize(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(
					device_to_query, m_surface, &format_count, details.formats.data());
		}

		uint32_t present_mode_count;
		vkGetPhysicalDeviceSurfacePresentModesKHR(
				device_to_query, m_surface, &present_mode_count, nullptr);
		if (present_mode_count != 0) {
			details.present_modes.resize(present_mode_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(
					device_to_query,
					m_surface,
					&present_mode_count,
					details.present_modes.data());
		}

		return details;
	}

	bool isDeviceSuitable(VkPhysicalDevice device_to_test){
		QueueFamilyIndices indices = findQueueFamilies(device_to_test);

		bool extensions_supported = checkDeviceExtensionSupport(device_to_test);

		bool swap_chain_adequate = false;
		if (extensions_supported){
			SwapChainSupportDetails swap_chain_support =
					querySwapChainSupport(device_to_test);
			swap_chain_adequate =
					!swap_chain_support.formats.empty() &&
					!swap_chain_support.present_modes.empty();
		}

		return indices.isComplete() && extensions_supported && swap_chain_adequate;
	}

	static bool checkDeviceExtensionSupport(VkPhysicalDevice device_to_check) {
		uint32_t extension_count = 0;
		vkEnumerateDeviceExtensionProperties(
				device_to_check, nullptr, &extension_count, nullptr);

		vector<VkExtensionProperties> available_extensions(extension_count);
		vkEnumerateDeviceExtensionProperties(
				device_to_check,
				nullptr,
				&extension_count,
				available_extensions.data());

		// local mutable set of required extension names
		set<string>required_extensions(
				device_extension_names.begin(), device_extension_names.end());

		// run through list of available.  Remove each from local mutable set
		for (const auto& extension : available_extensions) {
			required_extensions.erase(extension.extensionName);
		}

		// Local set will be empty, if each was found in list of available.
		return required_extensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device_to_check) {
		QueueFamilyIndices indices;  // stinking optionals hidden under the hood.

		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(
				device_to_check, &queue_family_count, nullptr);

		vector<VkQueueFamilyProperties>queue_families(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(
				device_to_check, &queue_family_count, queue_families.data());

		for (int i=0; i < queue_family_count; i++) {
			if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphics_family = i;
			}

			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(
					device_to_check, i, m_surface, &present_support);
			if (present_support) {
				indices.present_family = i;
			}

			if (indices.isComplete()) { // take the first queue that matches all needs
				break;
			}
		}
		return indices;
	}

	static vector<const char*> getRequiredExtensions() {
		uint32_t     glfw_extension_count = 0;
		const char **glfw_extensions;
		glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

		// Construction with what looks like horrible pointer math iterators *BB*
		vector<const char*> extensions(
				glfw_extensions, glfw_extensions + glfw_extension_count);

		if (enable_validation_layers) {
			// Going to need this extension to use validation layers
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // a.k.a VK_EXT_debug_utils
		}

		return extensions;
	}

	bool checkValidationLayerSupport(string& error_msg) {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		if (verbose_init) {
			listValidationLayers(availableLayers);
		}

		for (const char* layerName : validation_layer_names) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				error_msg = string(layerName) + ": validation layer requested, but not available!\n";
				return false;
			}
		}
		error_msg = "No Error";
		return true;
	}

	static void listValidationLayers(const vector<VkLayerProperties>& validation_layers) {

		uint32_t num_layers = validation_layers.size();
		cout << "Validation Layers: " << num_layers << endl;

		for (const auto& each_layer : validation_layers) {
			cout << "\t" << each_layer.layerName << endl;
		}
	}

	void printQueueFamilies(VkPhysicalDevice device_to_enumerate){
		QueueFamilyIndices indices;  // stinking optional hidden under the hood, so no need to initialize.  Isn't this easier - no need to initialize some vars.

		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(
				device_to_enumerate,
				&queue_family_count,
				nullptr);

		vector<VkQueueFamilyProperties>queue_families(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(
				device_to_enumerate,
				&queue_family_count,
				queue_families.data());

		cout << "Queue Family Count: " << queue_family_count << endl;
		for(int i=0; i< queue_family_count; i++) {
			bool has_bit = queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT;
			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(
					device_to_enumerate,
					i,
					m_surface,
					&present_support);
			cout << "\tQueueFamily[" << i << "] GRAPHICS_BIT ";
			cout << std::boolalpha << has_bit;
			cout << ", SurfaceSupport " << present_support << endl;
		}
	}
};

int main() {
	HelloTriangleApplication app = {};

	try {
		app.run();
	} catch (const std::exception& e)
	{
		std::cerr << e.what() << endl;
		return EXIT_FAILURE;
	}
	cout << "BB Main Normal Exit" << endl;
	return EXIT_SUCCESS;
}