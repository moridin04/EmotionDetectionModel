package com.example.emotiondetection

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.emotiondetection.ui.theme.EmotionDetectionTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Color
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {
    private lateinit var tflite: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, it)
            imageView.setImageBitmap(bitmap)
            val processedImage = preprocessImage(bitmap)
            val result = runInference(bitmap)
            resultText.text = "Prediction: $result"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)
        val btnSelect: Button = findViewById(R.id.btnSelect)

        tflite = Interpreter(loadModelFile("model4.tflite"))

        btnSelect.setOnClickListener {
            imagePicker.launch("image/*")
        }
    }

    private fun runInference(bitmap: Bitmap): String {
        val input = preprocessImage(bitmap)
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        tflite.run(input, output)

        val labels = loadLabels(this)
        val predictedIndex = output[0].indexOfMax()
        return labels.getOrElse(predictedIndex) { "Unknown" } +
                " (Confidence: ${"%.2f".format(output[0][predictedIndex] * 100)}%)"
    }

    private fun FloatArray.indexOfMax(): Int {
        return this.indices.maxByOrNull { this[it] } ?: -1
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 48, 48, true)

        val grayscale = Bitmap.createBitmap(48, 48, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscale)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(resizedBitmap, 0f, 0f, paint)

        val input = Array(1) { Array(48) { Array(48) { FloatArray(1) } } }
        for (x in 0 until 48) {
            for (y in 0 until 48) {
                val pixel = grayscale.getPixel(x, y)
                val r = Color.red(pixel)
                input[0][y][x][0] = r / 255.0f
            }
        }
        return input
    }

    companion object {
        const val NUM_CLASSES = 7
    }
}

fun loadLabels(context: Context): List<String> {
    return context.assets.open("labels.txt").bufferedReader().useLines { it.toList() }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        style = MaterialTheme.typography.titleLarge,
        modifier = modifier.padding(16.dp)
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    EmotionDetectionTheme {
        Greeting("Android")
    }
}
