package com.vulkan.compute;

import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

/**
 * MainActivity — shows a scrollable list of all Vulkan compute tutorial chapters.
 * Tapping a chapter launches VulkanChapterActivity which renders it via ANativeWindow.
 */
public class MainActivity extends AppCompatActivity {

    static { System.loadLibrary("vulkan_compute_chapters"); }

    public static native int      nativeChapterCount();
    public static native String[] nativeChapterInfo(int index); // [name, desc, available]

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        RecyclerView rv = findViewById(R.id.rv_chapters);
        rv.setLayoutManager(new LinearLayoutManager(this));
        rv.setAdapter(new ChapterAdapter());
    }

    private class ChapterAdapter extends RecyclerView.Adapter<ChapterAdapter.VH> {
        private final int count = nativeChapterCount();

        class VH extends RecyclerView.ViewHolder {
            TextView tvName, tvDesc;
            VH(View v) {
                super(v);
                tvName = v.findViewById(R.id.tv_chapter_name);
                tvDesc = v.findViewById(R.id.tv_chapter_desc);
            }
        }

        @Override
        public VH onCreateViewHolder(ViewGroup parent, int viewType) {
            View v = LayoutInflater.from(parent.getContext())
                    .inflate(R.layout.item_chapter, parent, false);
            return new VH(v);
        }

        @Override
        public void onBindViewHolder(VH holder, int pos) {
            String[] info = nativeChapterInfo(pos);
            holder.tvName.setText(info[0]);
            holder.tvDesc.setText(info[1]);
            boolean available = "true".equals(info[2]);
            holder.itemView.setAlpha(available ? 1.0f : 0.5f);
            holder.itemView.setOnClickListener(v -> {
                if (!available) return;
                Intent intent = new Intent(MainActivity.this, VulkanChapterActivity.class);
                intent.putExtra("CHAPTER_INDEX", pos);
                intent.putExtra("CHAPTER_NAME", info[0]);
                startActivity(intent);
            });
        }

        @Override
        public int getItemCount() { return count; }
    }
}
