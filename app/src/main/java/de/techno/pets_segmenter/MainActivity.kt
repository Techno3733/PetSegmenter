package de.techno.pets_segmenter

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import androidx.camera.lifecycle.ProcessCameraProvider
import android.util.Log
import androidx.camera.core.*
import de.techno.pets_segmenter.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var cameraSelector: CameraSelector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        // Set up the listeners for take photo and flip camera
        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }
        viewBinding.cameraFlipButton.setOnClickListener { flipCamera() }

        cameraExecutor = Executors.newSingleThreadExecutor()


        // this is just to show that the model works with real life images
        val catImageBitmap = BitmapFactory.decodeResource(resources, R.drawable.cat);
        val segmentedCat = segment(catImageBitmap)
        viewBinding.imageView.setImageBitmap(segmentedCat)
    }

    private var imageCapture: ImageCapture? = null

    // Select back camera as a default
    private fun startCamera(orientation: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            cameraSelector = orientation

            imageCapture = ImageCapture.Builder()
                .build()


            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, imageCapture, preview)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Set up image capture listener, which is triggered after photo has been taken
        imageCapture.takePicture(cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Image capture failed: ${exc.message}", exc)
                }

                override fun onCaptureSuccess(image: ImageProxy) {
                    super.onCaptureSuccess(image)
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.capacity())
                    buffer[bytes]
                    val bitmapImage = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    val rotation = image.imageInfo.rotationDegrees.toFloat()
                    image.close()

                    val segmentationImage = segment(bitmapImage)


                    // only UI Thread can touch the UI elements
                    Handler(Looper.getMainLooper()).post(Runnable {
                        viewBinding.imageView.rotation = rotation
                        viewBinding.imageView.setImageBitmap(segmentationImage)
                    })
                }
            })
    }

    private lateinit var interpreter: Interpreter
    private lateinit var model: MappedByteBuffer

    private fun segment(bitmap: Bitmap): Bitmap? {
        val originalHeight = bitmap.height
        val originalWidth = bitmap.width


        val imageWidth = 128
        val imageHeight = 128
        val numClasses = 3
        val bytesPerFloat = 4

        // Rescale image.
        val downscaledBitmap = Bitmap.createScaledBitmap(bitmap, imageWidth, imageHeight, false)
        val imgData: ByteBuffer = ByteBuffer.allocateDirect(bytesPerFloat * imageWidth * imageHeight * numClasses)
        imgData.order(ByteOrder.nativeOrder())
        imgData.rewind()

        var pixel = 0
        val intValues = IntArray(imageWidth * imageHeight)
        downscaledBitmap.getPixels(intValues, 0, imageWidth, 0, 0, imageWidth, imageHeight) // is stride correct?

        for (y in 0 until imageHeight) {
            for (x in 0 until imageWidth) {
                val value = intValues[pixel++]

                // Normalize channel values to [0.0, 1.0].
                imgData.putFloat(((value shr 16 and 0xFF) - 0.0f) / 255.0f) // r
                imgData.putFloat(((value shr 8 and 0xFF) - 0.0f) / 255.0f) // g
                imgData.putFloat(((value and 0xFF) - 0.0f) / 255.0f) // b
            }
        }
        imgData.rewind()

        try {
            val options = Interpreter.Options()
            options.numThreads = 4
            model = loadModelFile(assets, "model.tflite")
            interpreter = Interpreter(model, options)
        } catch (e: java.lang.Exception) {
            throw RuntimeException(e)
        }

        val colors = arrayOf(Color.RED, Color.TRANSPARENT, Color.GREEN)

        val output: ByteBuffer = ByteBuffer.allocateDirect(bytesPerFloat * imageWidth * imageHeight * numClasses)
        output.order(ByteOrder.nativeOrder())
        Log.d("Progress:", "Start interpreting ...")
        interpreter.run(imgData, output)
        Log.d("Progress:", "Finished interpreting")
        //Log.d("Output", output.toString())

        val mSegmentBits = Array(imageWidth) { IntArray(imageHeight) }

        val imageBitmap: Bitmap = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888)

        output.rewind()

        // argmax in Kotlin
        for (y in 0 until imageHeight) {
            for (x in 0 until imageWidth) {
                var maxVal = 0f
                mSegmentBits[x][y] = 0

                // determine the most likely class
                for (c in 0 until numClasses) {
                    val value =
                        output.getFloat((y * imageWidth * numClasses + x * numClasses + c) * 4)

                    // always guess class 0 first, and the look if something else is more likely
                    if (c == 0 || value > maxVal) {
                        maxVal = value
                        mSegmentBits[x][y] = c
                    }
                }

                // put in the color of the most likely class
                imageBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])
            }
        }

        // scale image back from (320, 320) to original size
        val upscaledBitmap = Bitmap.createScaledBitmap(imageBitmap, originalWidth, originalHeight, false)

        return overlay(bitmap, upscaledBitmap, 100)
    }

    private fun overlay(bmp1: Bitmap, bmp2: Bitmap, opacity: Int): Bitmap? {
        val bmOverlay = Bitmap.createBitmap(bmp1.width, bmp1.height, bmp1.config)
        val canvas = Canvas(bmOverlay)

        // add Alpha
        for (x in 0 until bmp2.width){
            for (y in 0 until bmp2.height){
                val originalColor = bmp2.getPixel(x,y)
                val colorWithoutAlpha = originalColor.shl(8).shr(8)
                val colorWithCustomAlpha = colorWithoutAlpha + opacity.and(0xFF).shl(24)
                bmp2.setPixel(x,y,colorWithCustomAlpha)
            }
        }

        canvas.drawBitmap(bmp1, Matrix(), null)
        canvas.drawBitmap(bmp2, Matrix(), null)
        return bmOverlay
    }

    private fun flipCamera() {
        if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
            startCamera(CameraSelector.DEFAULT_FRONT_CAMERA)
        } else {
            startCamera(CameraSelector.DEFAULT_BACK_CAMERA)
        }
    }

    // my own simple permission request function
    private fun requestPermissions() {
        ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS,  REQUEST_CODE_ASK_PERMISSION)

        if (allPermissionsGranted())
            Log.d("Test", "REACHED!!!!!!!!!!!!!!!!!!!!!")
            startCamera()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf (
                Manifest.permission.CAMERA,
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
        private const val REQUEST_CODE_ASK_PERMISSION = 1;
    }


}