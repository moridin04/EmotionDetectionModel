package com.example.emotiondetection

import android.app.ProgressDialog
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var tflite: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var grayscaleView: ImageView
    private lateinit var resultText: TextView
    private lateinit var progressDialog: ProgressDialog

    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                // Show loading dialog
                progressDialog.show()

                // Load the original image with sampling
                val options = BitmapFactory.Options().apply {
                    inSampleSize = calculateInSampleSize(it, 800, 800)
                }
                val originalBitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)

                // Convert to grayscale
                val grayscaleBitmap = convertToGrayscale(originalBitmap)

                // Display original image
                imageView.setImageBitmap(originalBitmap)

                // Display grayscale preview (ADD THESE LINES)
                findViewById<FrameLayout>(R.id.grayscaleContainer).visibility = View.VISIBLE
                grayscaleView.setImageBitmap(grayscaleBitmap)

                // Hide placeholder text
                findViewById<TextView>(R.id.placeholderText).visibility = View.GONE

                // Run inference on grayscale image
                val result = runInference(grayscaleBitmap)
                resultText.text = "Prediction: $result"

            } catch (e: Exception) {
                // Hide grayscale preview if error occurs (ADD THIS LINE)
                findViewById<FrameLayout>(R.id.grayscaleContainer).visibility = View.GONE
                Toast.makeText(this, "Error processing image", Toast.LENGTH_SHORT).show()
                e.printStackTrace()
            } finally {
                progressDialog.dismiss()
            }
        } ?: run {
            // Hide grayscale preview if no image selected (ADD THIS LINE)
            findViewById<FrameLayout>(R.id.grayscaleContainer).visibility = View.GONE
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize views
        imageView = findViewById(R.id.imageView)
        grayscaleView = findViewById(R.id.grayscaleView)
        val grayscaleContainer = findViewById<FrameLayout>(R.id.grayscaleContainer)
        resultText = findViewById(R.id.resultText)
        val btnSelect: Button = findViewById(R.id.btnSelect)

        // Initialize progress dialog
        progressDialog = ProgressDialog(this).apply {
            setMessage("Processing image...")
            setCancelable(false)
        }

        grayscaleContainer.visibility = View.GONE

        // Initialize TensorFlow Lite
        tflite = Interpreter(loadModelFile("model4.tflite"))

        btnSelect.setOnClickListener {
            imagePicker.launch("image/*")
        }
    }

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        val grayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscale)
        val paint = Paint()

        val colorMatrix = ColorMatrix().apply {
            setSaturation(0f) // Convert to grayscale
        }

        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)

        return grayscale
    }

    private fun calculateInSampleSize(uri: Uri, reqWidth: Int, reqHeight: Int): Int {
        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        }

        val (height: Int, width: Int) = options.run { outHeight to outWidth }
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2

            while (halfHeight / inSampleSize >= reqHeight &&
                halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }

        return inSampleSize
    }

    private fun runInference(bitmap: Bitmap): String {
        val input = preprocessImage(bitmap)
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        tflite.run(input, output)

        val labels = loadLabels(this)
        val predictedIndex = output[0].indexOfMax()
        return labels.getOrElse(predictedIndex) { "Unknown" }
    }

    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 48, 48, true)

        val input = Array(1) { Array(48) { Array(48) { FloatArray(1) } } }
        for (x in 0 until 48) {
            for (y in 0 until 48) {
                val pixel = resizedBitmap.getPixel(x, y)
                val r = Color.red(pixel) // Grayscale so R=G=B
                input[0][y][x][0] = r / 255.0f
            }
        }
        return input
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

    companion object {
        const val NUM_CLASSES = 7
    }
}

fun loadLabels(context: Context): List<String> {
    return context.assets.open("labels.txt").bufferedReader().useLines { it.toList() }
}