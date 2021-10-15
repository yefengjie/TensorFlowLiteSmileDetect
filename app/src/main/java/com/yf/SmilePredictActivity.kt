package com.yf

import android.annotation.SuppressLint
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Size
import android.view.MenuItem
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.yf.databinding.ActivitySmilePredictBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import java.nio.ByteBuffer

class SmilePredictActivity : AppCompatActivity() {

    /**
     * 深度学习Api是否初始化
     */
    private var nnapiInitialized = false

    /**
     * 深度学习Api委托器
     */
    private val nnApiDelegate by lazy {
        NnApiDelegate().apply { nnapiInitialized = true }
    }

    /**
     * tensor flow lite
     */
    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, "lenet.tflite"),
            Interpreter.Options().addDelegate(nnApiDelegate)
        )
    }

    /**
     * 模型输入尺寸
     */
    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    private lateinit var mBinding: ActivitySmilePredictBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_smile_predict)
        mBinding = ActivitySmilePredictBinding.inflate(layoutInflater)
        setContentView(mBinding.root)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "笑脸检测"

        mBinding.btnStart.setOnClickListener {
            start(R.mipmap.normal, mBinding.ivPreprocessNormal, mBinding.tvResultNormal)
            start(R.mipmap.smiles, mBinding.ivPreprocessSmile, mBinding.tvResultSmile)
            TfLiteGpuCompare.start(this, mBinding.tvLog)
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        if (item.itemId == android.R.id.home) {
            finish()
            return true
        }
        return super.onOptionsItemSelected(item)
    }

    override fun onDestroy() {
        if (nnapiInitialized) nnApiDelegate.close()
        super.onDestroy()
    }

    /**
     * 开始推理
     * @param id 原始图片资源id
     * @param processedImage 预处理后的图片
     * @param tv 输出结果文本
     */
    @SuppressLint("SetTextI18n")
    private fun start(id: Int, processedImage: ImageView, tv: TextView) {
        val inputBuffer = processImgToInput(id, processedImage)
        val outputBuffer = mapOf(0 to arrayOf(FloatArray(2)))
        val timeStart = System.currentTimeMillis()
        tflite.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputBuffer)
        val predictTime = System.currentTimeMillis() - timeStart
        val outputArray = (outputBuffer[0] as Array)[0]
        val result = if (outputArray[0] > outputArray[1]) "正常" else "笑脸"
        tv.text = "预测结果: $result -> 推理耗时:${predictTime}ms"
    }

    /**
     * 预处理图片，并转换为输入
     * @param id 原始图片资源id
     * @param processedImage 预处理后的图片
     */
    private fun processImgToInput(id: Int, processedImage: ImageView): ByteBuffer {
        val tmpBitmap = BitmapFactory.decodeResource(resources, id)
        // 输入的图片可能不是正方形，我们需要按照短边裁剪
        val cropSize = minOf(tmpBitmap.width, tmpBitmap.height)
        val imgProcessor = ImageProcessor.Builder()
            //按cropSize对图片进行裁剪
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            //将裁剪好的图片缩放到输入尺寸
            .add(
                ResizeOp(
                    tfInputSize.height,
                    tfInputSize.width,
                    ResizeOp.ResizeMethod.NEAREST_NEIGHBOR
                )
            )
            //把像素值的范围进行归一化从[0,255]区间映射到[-1,1]区间
            .add(NormalizeOp(0f, 1f))
            // 转化为灰度图
            .add(TransformToGrayscaleOp())
            .build()
        val tfImageBuffer = TensorImage(DataType.FLOAT32)
        val tfImage = imgProcessor.process(tfImageBuffer.apply { load(tmpBitmap) })
        processedImage.setImageBitmap(tfImage.bitmap)
        return tfImage.buffer
    }
}