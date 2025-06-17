package com.example.emotiondetection

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
    // TensorFlow Life interpreter
    private lateinit var tflite: Interpreter

    // UI Components
    private lateinit var imageView: ImageView
    private lateinit var grayscaleView: ImageView
    private lateinit var resultText: TextView

    // Launches Gallery for Image Picking
    private val imagePicker = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                // Target Dimensions for Image
                val reqWidth = 48
                val reqHeight = 48

                // Sampling size to load smaller version of image
                val options = BitmapFactory.Options().apply {
                    inSampleSize = calculateInSampleSize(it, reqWidth, reqHeight)
                    inJustDecodeBounds = false
                }

                // Loading the original bitmap with calculated options
                val inputStream = contentResolver.openInputStream(it)
                val originalBitmap = BitmapFactory.decodeStream(inputStream, null, options)
                inputStream?.close()

                // Crop and resizing to 48x48, then converting to grayscale
                val croppedResized = centerCropAndResize(originalBitmap!!, 48)
                val grayscaleBitmap = convertToGrayscale(croppedResized)

                // Displaying the processed image
                imageView.setImageBitmap(grayscaleBitmap)
                findViewById<TextView>(R.id.placeholderText).visibility = View.GONE

                // Detecting emotion in image, then displaying the prediction result
                val result = runInference(grayscaleBitmap)
                resultText.text = "Prediction: $result"

            } catch (e: Exception) {
                findViewById<FrameLayout>(R.id.grayscaleContainer).visibility = View.GONE
                Toast.makeText(this, "Error processing image", Toast.LENGTH_SHORT).show()
                e.printStackTrace()
            }
        } ?: run {
            // Hiding grayscale container if no image was selected
            findViewById<FrameLayout>(R.id.grayscaleContainer).visibility = View.GONE
        }
    }

    // Cropping and resizing the uploaded image
    private fun centerCropAndResize(bitmap: Bitmap, size: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        // Crop dimensions (Square)
        val newEdge = minOf(width, height)
        val xOffset = (width - newEdge) / 2
        val yOffset = (height - newEdge) / 2

        // Resizing cropped bitmap
        val croppedBitmap = Bitmap.createBitmap(bitmap, xOffset, yOffset, newEdge, newEdge)
        return Bitmap.createScaledBitmap(croppedBitmap, size, size, true)
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // UI Components
        imageView = findViewById(R.id.imageView)
        grayscaleView = findViewById(R.id.grayscaleView)
        val grayscaleContainer = findViewById<FrameLayout>(R.id.grayscaleContainer)
        resultText = findViewById(R.id.resultText)
        val btnSelect: Button = findViewById(R.id.btnSelect)

        grayscaleContainer.visibility = View.GONE

        // Loading the TensorFlow Lite model
        tflite = Interpreter(loadModelFile("emotionmodel.tflite"))

        // Set click listener for select button
        btnSelect.setOnClickListener {
            imagePicker.launch("image/*")
        }
    }

    // Convert the bitmap to grayscale
    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        // Creating new bitmap for grayscale output
        val grayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayscale)
        val paint = Paint()

        // Creating color matrix to remove saturation and make it grayscale
        val colorMatrix = ColorMatrix().apply {
            setSaturation(0f)
        }

        // Applying the color matrix filter
        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)

        return grayscale
    }

    // Calculating optimal sampling size to load smaller version of the image
    private fun calculateInSampleSize(uri: Uri, reqWidth: Int, reqHeight: Int): Int {
        val options = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        }

        // Calculating largest inSampleSize value
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

    // Runnign Emotion Detection inference on processed image
    private fun runInference(bitmap: Bitmap): String {
        // Preprocessing image for the model
        val input = preprocessImage(bitmap)
        // Array to store model predictions
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        // Run model
        tflite.run(input, output)

        // Load labels.txt
        val labels = loadLabels(this)
        val probabilities = output[0]

        // Temperature scaling to soften probabilities
        val temperature = 2.0f
        val softenedProbs = probabilities.map {
            (it / temperature).coerceAtLeast(0f)
        }.toFloatArray()

        // Normalize probabilities to sum to 1
        val sum = softenedProbs.sum()
        val normalizedProbs = softenedProbs.map { it / sum }.toFloatArray()

        // Find predicted emotion with highest probability
        val predictedIndex = normalizedProbs.indexOfMax()
        val confidence = normalizedProbs[predictedIndex]

        // Only return prediction if confidence exceeds threshold
        val threshold = 0.01f
        return if (confidence > threshold) {
            labels.getOrElse(predictedIndex) { "Unknown" }
        } else {
            "Uncertain (${labels.getOrElse(predictedIndex) { "Unknown" }}: ${"%.1f".format(confidence * 100)}%)"
        }
    }

    // Preprocessing of image for TensorFlow Lite model
    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        // Ensure image is 48x48
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 48, 48, true)

        // Creating a 4D array (1, 48, 48, 1) for model input
        val input = Array(1) { Array(48) { Array(48) { FloatArray(1) } } }
        for (x in 0 until 48) {
            for (y in 0 until 48) {
                // Get pixel value and normalize to [0, 1]
                val pixel = resizedBitmap.getPixel(x, y)
                val r = Color.red(pixel)
                input[0][y][x][0] = r / 255.0f
            }
        }
        return input
    }

    // Function for finding index of maximum value in FloatArray
    private fun FloatArray.indexOfMax(): Int {
        return this.indices.maxByOrNull { this[it] } ?: -1
    }

    // Function for loading TensorFlow Lite model from assets
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

// Loading emotion labels from assets folder
fun loadLabels(context: Context): List<String> {
    return context.assets.open("labels.txt").bufferedReader().useLines { it.toList() }
}