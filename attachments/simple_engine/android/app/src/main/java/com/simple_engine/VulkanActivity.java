package com.simple_engine;

import android.os.Bundle;
import android.view.WindowManager;
import com.google.androidgamesdk.GameActivity;

public class VulkanActivity extends GameActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    static {
        System.loadLibrary("simple_engine_android");
    }
}
