package com.example.emotiondetection

import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MakersActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_makers)

        findViewById<Button>(R.id.btnBack).setOnClickListener {
            finish() // Return to choice screen
        }
    }
}