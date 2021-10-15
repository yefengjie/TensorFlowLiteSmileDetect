package com.yf

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.BitmapFactory
import android.util.Size
import android.widget.TextView
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import java.nio.ByteBuffer

/**
 * tflite 一次性推理测试
 */
@SuppressLint("SetTextI18n")
object TfLiteGpuCompare {

    fun start(context: Context, logTv: TextView) {
        logTv.text = ""
        doStart(context, false, logTv)
        doStart(context, true, logTv)
    }

    private fun doStart(
        context: Context,
        isGpu: Boolean,
        logTv: TextView
    ) {
        Thread.sleep(100)
        // 委托器
        val delegate = if (isGpu) GpuDelegate() else NnApiDelegate()
        // 初始化tflite
        val tflite = Interpreter(
            FileUtil.loadMappedFile(context, "lenet.tflite"),
            Interpreter.Options().addDelegate(delegate)
        )
        // 输入维度
        val inputShape = tflite.getInputTensor(0).shape()
        // 输入维度转为size
        val tfInputSize = Size(inputShape[2], inputShape[1])
        // 预处理图片转为输入
        val inputBuffer = processImgToInput(context, tfInputSize)
        // 定义输出
        val outputBuffer = mapOf(0 to arrayOf(FloatArray(2)))
        // 开始推理时间打点
        val timeStart = System.currentTimeMillis()
        // 开始推理
        tflite.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputBuffer)
        // 计算推理耗时
        val predictTime = System.currentTimeMillis() - timeStart
        // 取出推理结果
        val outputArray = (outputBuffer[0] as Array)[0]
        // 将推理结果转为显示文案
        val result = if (outputArray[0] > outputArray[1]) "正常" else "笑脸"
        // 显示文案
        val resultTxt = "${if (isGpu) "Gpu" else "NnApi"} 预测结果:$result -> 耗时:${predictTime}ms"
        logTv.text = logTv.text.toString() + "\n" + resultTxt
        // 关闭
        delegate.close()
    }

    /**
     * 预处理图片，并转换为输入
     * @param inSize 输入维度
     */
    private fun processImgToInput(context: Context, inSize: Size): ByteBuffer {
        val tmpBitmap = BitmapFactory.decodeResource(context.resources, R.mipmap.smiles)
        // 输入的图片可能不是正方形，我们需要按照短边裁剪
        val cropSize = minOf(tmpBitmap.width, tmpBitmap.height)
        val imgProcessor = ImageProcessor.Builder()
            //按cropSize对图片进行裁剪
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            //将裁剪好的图片缩放到输入尺寸
            .add(ResizeOp(inSize.height, inSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            //把像素值的范围进行归一化从[0,255]区间映射到[-1,1]区间
            .add(NormalizeOp(0f, 1f))
            // 转化为灰度图
            .add(TransformToGrayscaleOp())
            .build()
        val tfImageBuffer = TensorImage(DataType.FLOAT32)
        val tfImage = imgProcessor.process(tfImageBuffer.apply { load(tmpBitmap) })
        tmpBitmap.recycle()
        return tfImage.buffer
    }
}