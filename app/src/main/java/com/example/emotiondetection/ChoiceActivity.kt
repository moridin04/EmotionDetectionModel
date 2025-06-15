package com.example.emotiondetection

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class ChoiceActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_choice)

        // Button to Model Detection
        findViewById<Button>(R.id.btnModel).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }

        // Button to Makers screen
        findViewById<Button>(R.id.btnMakers).setOnClickListener {
            startActivity(Intent(this, MakersActivity::class.java))
        }
    }
}