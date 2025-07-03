package com.vulkan.tutorial;

import android.os.Bundle;
import android.view.WindowManager;
import com.google.androidgamesdk.GameActivity;

public class VulkanActivity extends GameActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Keep the screen on while the app is running
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    // Load the native library
    static {
        System.loadLibrary("vulkan_tutorial_android");
    }
}
