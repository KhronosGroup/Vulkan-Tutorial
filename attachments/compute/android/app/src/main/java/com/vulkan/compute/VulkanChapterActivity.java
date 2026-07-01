package com.vulkan.compute;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import androidx.appcompat.app.AppCompatActivity;

/**
 * VulkanChapterActivity — holds a full-screen SurfaceView and runs the selected
 * chapter's Vulkan renderer on a dedicated background thread.
 *
 * Touch input is translated to GLFW-style events via the native event queue in
 * glfw_android_shim.h so the chapter's existing callback handlers work unchanged.
 */
public class VulkanChapterActivity extends AppCompatActivity implements SurfaceHolder.Callback {

    // JNI — implemented in android_host.cpp
    public native void nativeStart(Surface surface, AssetManager mgr, int chapterIndex);
    public native void nativeStop();
    public native void nativeTouchCursorPos(double x, double y);
    public native void nativeMouseButton(int button, int action, double x, double y);
    public native void nativeScroll(double dx, double dy);
    public native void nativeResize(int w, int h);

    private static final int GLFW_MOUSE_BUTTON_LEFT = 0;
    private static final int GLFW_PRESS             = 1;
    private static final int GLFW_RELEASE           = 0;

    private Thread  mRenderThread;
    private Surface mSurface;
    private int     mChapterIndex;
    private float   mPrevPinchDist = -1f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chapter);

        mChapterIndex = getIntent().getIntExtra("CHAPTER_INDEX", 0);
        String name   = getIntent().getStringExtra("CHAPTER_NAME");
        setTitle(name != null ? name : "Vulkan Compute");

        SurfaceView sv = findViewById(R.id.sv_vulkan);
        sv.getHolder().addCallback(this);

        sv.setOnTouchListener((v, ev) -> {
            int count = ev.getPointerCount();
            switch (ev.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    nativeMouseButton(GLFW_MOUSE_BUTTON_LEFT, GLFW_PRESS,
                            ev.getX(), ev.getY());
                    break;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    nativeMouseButton(GLFW_MOUSE_BUTTON_LEFT, GLFW_RELEASE,
                            ev.getX(), ev.getY());
                    mPrevPinchDist = -1f;
                    break;
                case MotionEvent.ACTION_MOVE:
                    if (count == 1) {
                        nativeTouchCursorPos(ev.getX(), ev.getY());
                    } else if (count >= 2) {
                        // Two-finger pinch maps to scroll (zoom)
                        float dx = ev.getX(0) - ev.getX(1);
                        float dy = ev.getY(0) - ev.getY(1);
                        float dist = (float) Math.sqrt(dx * dx + dy * dy);
                        if (mPrevPinchDist > 0f) {
                            float delta = (dist - mPrevPinchDist)
                                    / sv.getHeight() * 10.0f;
                            nativeScroll(0.0, delta);
                        }
                        mPrevPinchDist = dist;
                    }
                    break;
                default:
                    break;
            }
            return true;
        });
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        mSurface = holder.getSurface();
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int fmt, int w, int h) {
        if (mRenderThread == null || !mRenderThread.isAlive()) {
            mSurface = holder.getSurface();
            final AssetManager mgr = getAssets();
            final int idx = mChapterIndex;
            mRenderThread = new Thread(() -> nativeStart(mSurface, mgr, idx), "VulkanRender");
            mRenderThread.start();
        } else {
            nativeResize(w, h);
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        nativeStop();
        try {
            if (mRenderThread != null) mRenderThread.join(3000);
        } catch (InterruptedException ignored) {}
        mSurface = null;
    }

    @Override
    protected void onDestroy() {
        nativeStop();
        super.onDestroy();
    }
}
