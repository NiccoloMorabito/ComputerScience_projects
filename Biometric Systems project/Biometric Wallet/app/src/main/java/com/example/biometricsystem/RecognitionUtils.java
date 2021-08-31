package com.example.biometricsystem;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class RecognitionUtils {
    private android.graphics.Rect faceRect;
    private FaceDetectorOptions realTimeOpts;
    private TFLiteObjectDetectionAPIModel detector;
    private static final int TF_OD_API_INPUT_SIZE = 112;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private double threshold=0.9;
    private int counter = 0;

    public RecognitionUtils(Context context){
        faceRect = new android.graphics.Rect();
        realTimeOpts = new FaceDetectorOptions.Builder()
                .setLandmarkMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .build();
        try {
            detector =(TFLiteObjectDetectionAPIModel) TFLiteObjectDetectionAPIModel.create(
                context.getAssets(),
                TF_OD_API_MODEL_FILE,
                TF_OD_API_LABELS_FILE,
                TF_OD_API_INPUT_SIZE,
                TF_OD_API_IS_QUANTIZED
            );
        } catch (final IOException e) {
            e.printStackTrace();
            System.out.println("Classifier could not be initialized");
        }
    }

    private Bitmap getGreyscaleBitmapFrom(InputStream imageStream) {
        Bitmap original = BitmapFactory.decodeStream(imageStream);
        return toGrayscale(original);
    }

    public boolean hasFace(InputStream imageStream) {
        Bitmap originalBmp = produceOriginalBitmap(imageStream);
        return getMaxFace(originalBmp)!=null;
    }

    public void saveFacesDetectedIn(InputStream imageStream, String directory) throws ExecutionException, InterruptedException, FileNotFoundException {
        Bitmap originalBmp = produceOriginalBitmap(imageStream);
        List<Face> faces = getFacesIn(originalBmp);
        for (Face face : faces) {
            Bitmap croppedFaceBmp = getCroppedFaceBmp(face, originalBmp);
            saveBitmap(croppedFaceBmp, directory);
        }
    }

    private Bitmap produceOriginalBitmap(InputStream imageStream) {
        Bitmap bg = getGreyscaleBitmapFrom(imageStream);
        Mat mGrey = new Mat();
        Bitmap bmp32 = bg.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, mGrey);
        final Bitmap originalBmp = Bitmap.createBitmap(mGrey.cols(), mGrey.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(mGrey, originalBmp);
        return originalBmp;
    }

    private Face getMaxFace(Bitmap originalBmp) {
        try {
            List<Face> faces = getFacesIn(originalBmp);
            if (faces.size() > 0) {
                Face maxFace = faces.get(0);
                for (Face f : faces) {
                    if (f.getBoundingBox().width() * f.getBoundingBox().height() >
                            maxFace.getBoundingBox().width() * maxFace.getBoundingBox().height()) {
                        maxFace = f;
                    }
                }
                return maxFace;
            }
            return null;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private List<Face> getFacesIn(Bitmap originalBmp) throws ExecutionException, InterruptedException {
        final InputImage image = InputImage.fromBitmap(originalBmp, 0);
        final FaceDetector detector = FaceDetection.getClient(realTimeOpts);
        return Tasks.await(detector.process(image));
    }

    public void enroll(String name, InputStream imageStream) throws NoFaceDetectedException {
        Bitmap originalBmp = produceOriginalBitmap(imageStream);
        Face maxFace = getMaxFace(originalBmp);
        if (maxFace == null)
            throw new NoFaceDetectedException(String.format("Enrollment failed for %s: no face has been detected in the image", name));

        Bitmap croppedFaceBmp = getCroppedFaceBmp(maxFace, originalBmp);

        List<SimilarityClassifier.Recognition> results = this.detector.recognizeImage(croppedFaceBmp, true);
        SimilarityClassifier.Recognition enrollmentTarget=new SimilarityClassifier.Recognition("0",name,-1f,new RectF(faceRect));
        enrollmentTarget.setCrop(croppedFaceBmp);
        enrollmentTarget.setExtra(results.get(0).getExtra());
        this.detector.register(name,enrollmentTarget);
    }

    /**
     * Verificate a person from an image
     * @param name of the person
     * @param imageStream
     * @return the distance from the person with that name if imageStream contains a face
     * @throws Exception in case no face has been detected
     */
    public double verificate(String name, InputStream imageStream) throws NoFaceDetectedException {
        Bitmap originalBmp = produceOriginalBitmap(imageStream);
        Face maxFace = getMaxFace(originalBmp);
        if (maxFace == null)
            throw new NoFaceDetectedException(String.format("Verification failed for %s: no face has been detected in the image", name));
        Bitmap croppedFaceBmp = getCroppedFaceBmp(maxFace, originalBmp);
        return this.detector.verificate(croppedFaceBmp, name);
    }

    /**
     * If a person is verified
     * @param name of the person
     * @param imageStream
     * @return true in case imageStream contains a face with a distance from the real one less than
     * the threshold, false otherwise
     * @throws Exception in case no face has been detected
     */
    public boolean isVerified(String name, InputStream imageStream) throws NoFaceDetectedException {
        double distance = verificate(name, imageStream);
        if (distance == -1 || distance > this.threshold)
            return false;
        return true;
    }

    private int[] validateCoordinate(int left, int top, int right, int bottom, Bitmap bitmap) {
        int[]coordinate=new int[4];
        if(left<0){left=0;}
        if(top<0){top=0;}
        if(right>bitmap.getWidth()){right=bitmap.getWidth();}
        if(bottom>bitmap.getHeight()){bottom=bitmap.getHeight();}
        coordinate[0]=left;
        coordinate[1]=top;
        coordinate[2]=right;
        coordinate[3]=bottom;
        return coordinate;
    }

    public String recognizeFromImage(InputStream imageStream) throws NoFaceDetectedException {
        Bitmap originalBmp = produceOriginalBitmap(imageStream);
        Face maxFace = getMaxFace(originalBmp);
        if (maxFace == null)
            throw new NoFaceDetectedException(String.format("Recognition failed since no face has been detected in the image"));

        Bitmap croppedFaceBmp = getCroppedFaceBmp(maxFace, originalBmp);

        List<SimilarityClassifier.Recognition> results = this.detector.recognizeImage(croppedFaceBmp, false);
        final SimilarityClassifier.Recognition re = results.get(0);
        if (re.getDistance()<threshold)
            return re.getTitle();
        return "Unknown";
    }

    public boolean recognize(String name, InputStream imageStream) throws Exception {
        if (recognizeFromImage(imageStream).equals(name))
            return true;
        return false;
    }

    private Bitmap getCroppedFaceBmp(Face maxFace, Bitmap originalBmp) {
        android.graphics.Rect r = maxFace.getBoundingBox();
        int[] coordinate=validateCoordinate(r.left,r.top,r.right,r.bottom,originalBmp);
        faceRect.set(coordinate[0],coordinate[1],coordinate[2],coordinate[3]);
        Bitmap faceBmp = originalBmp;
        Bitmap croppedFaceBmp = Bitmap.createBitmap(faceBmp, faceRect.left, faceRect.top, Math.abs(faceRect.width()), Math.abs(faceRect.height()));
        return Bitmap.createScaledBitmap(croppedFaceBmp, 112, 112, true);
    }

    private void saveBitmap(Bitmap bmp, String directory) throws FileNotFoundException {
        String filePath = directory + "/" + counter++ + ".png";
        System.out.println(filePath);
        FileOutputStream out = new FileOutputStream(filePath);
        bmp.compress(Bitmap.CompressFormat.PNG, 100, out);
    }

    private Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();
        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    public void setThreshold(double threshold) {
        this.threshold=threshold;
    }

}
