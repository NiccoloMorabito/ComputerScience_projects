package com.example.biometricsystem;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.RectF;
import android.os.Bundle;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.PortraitCameraView;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import android.app.AlertDialog;

import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.firebase.FirebaseApp;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import static org.opencv.core.CvType.CV_8U;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    JavaCameraView javaCameraView;
    File cascadeFile;
    CascadeClassifier faceDetector;
    private Mat mRgba,mGrey;
    // MobileFaceNet
    private static final int TF_OD_API_INPUT_SIZE = 112;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";


    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private int cameraIndex;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    //private static final int CROP_SIZE = 320;
    //private static final Size CROP_SIZE = new Size(320, 320);


    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    private Integer sensorOrientation;

    private TFLiteObjectDetectionAPIModel detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;
    private boolean addPending = false;
    //private boolean adding = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    //private Matrix cropToPortraitTransform;


    public boolean detected;
    // here the face is cropped and drawn
    private Bitmap faceBmp = null;
    private Bitmap croppedFaceBmp;
    private  Bitmap croppedFaceBmp2;
    android.graphics.Rect faceRect;
    private Button fabAdd;
    FaceDetectorOptions realTimeOpts;

    private void setVisibility(boolean vis){
        if(vis){findViewById(R.id.addImage).setVisibility(View.VISIBLE);}
        else{findViewById(R.id.addImage).setVisibility(View.GONE);}
    }
    private void swapCamera() {

        if(cameraIndex==0){cameraIndex=1;}
        else{cameraIndex=0;}
        javaCameraView.disableView();
        javaCameraView.setCameraIndex(cameraIndex);
        javaCameraView.enableView();

        }
    @Override
    public View onCreateView(String name, Context context, AttributeSet attrs) {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        faceRect = new android.graphics.Rect();

        return super.onCreateView(name, context, attrs);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        javaCameraView=(JavaCameraView) findViewById(R.id.javacameraview);
        cameraIndex=0;
        javaCameraView.setCameraIndex(0);
        // Real-time contour detection of multiple faces
        realTimeOpts = new FaceDetectorOptions.Builder()
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .build();


        if(!OpenCVLoader.initDebug())
        {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,callback);
        }
        else{
            callback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        javaCameraView.setCvCameraViewListener(this);

        fabAdd=findViewById(R.id.addImage);

        fabAdd.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                    //Questo è meglio se lo istanzi dopo solo quando ti serve per la registrazione. Passa in input le due mappe.
                    String label = "";
                    float confidence = -1f;
                    Integer color = Color.BLUE;
                    Object extra = null;
                    Bitmap crop = null;
                    SimilarityClassifier.Recognition rec = new SimilarityClassifier.Recognition("0", label, confidence, new RectF(faceRect));
                    rec.setCrop(croppedFaceBmp);
                    rec.setRgba(croppedFaceBmp2);
                    showAddFaceDialog(rec);
            }
        });
        Button switchCamera=(Button) findViewById(R.id.switch_camera);
        switchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                swapCamera();
            }
        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba=new Mat();
        mGrey=new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGrey.release();
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame)  {
        mRgba=inputFrame.rgba();
        mGrey=inputFrame.gray();
        MatOfRect faceDetections=new MatOfRect();
        Bitmap originalBmp= Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(mRgba,originalBmp);
        InputImage image = InputImage.fromBitmap(originalBmp,0);
        FaceDetector detector = FaceDetection.getClient(realTimeOpts);
        try {

            List<Face> authResult = Tasks.await(
                    detector.process(image));

            if (authResult.size() > 0) {

                Face maxFace=authResult.get(0);
                for(Face f:authResult)
                {
                    if(f.getBoundingBox().width()*f.getBoundingBox().height()>
                            maxFace.getBoundingBox().width()*maxFace.getBoundingBox().height()) {maxFace=f;}
                }
                android.graphics.Rect r = maxFace.getBoundingBox();
                faceRect.set(r.left, r.top, r.right, r.bottom);
                faceBmp=Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mGrey,faceBmp);
                croppedFaceBmp = Bitmap.createBitmap(faceBmp, faceRect.left, faceRect.top, Math.abs(faceRect.width()),Math.abs(faceRect.height()));
                croppedFaceBmp=Bitmap.createScaledBitmap(croppedFaceBmp, 112, 112, true);
                faceBmp=Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.RGB_565);
                Utils.matToBitmap(mRgba,faceBmp);
                croppedFaceBmp2 = Bitmap.createBitmap(faceBmp, faceRect.left, faceRect.top, Math.abs(faceRect.width()),Math.abs(faceRect.height()));
                croppedFaceBmp2=Bitmap.createScaledBitmap(croppedFaceBmp2, 112, 112, true);
                Imgproc.rectangle(mRgba, new Point(r.left, r.top), new Point(r.right, r.bottom), new Scalar(255, 0, 0));
                detected=true;
            }
            else{
                detected=false;
            }

        }
        catch (Exception e){}
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                setVisibility(detected);
            }});

        return mRgba;
    }

    private BaseLoaderCallback callback=new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            try {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS: {

                        javaCameraView.enableView();
                        try {
                            detector =(TFLiteObjectDetectionAPIModel)
                                    TFLiteObjectDetectionAPIModel.create(
                                            getAssets(),
                                            TF_OD_API_MODEL_FILE,
                                            TF_OD_API_LABELS_FILE,
                                            TF_OD_API_INPUT_SIZE,
                                            TF_OD_API_IS_QUANTIZED);
                            //cropSize = TF_OD_API_INPUT_SIZE;
                        } catch (final IOException e) {
                            e.printStackTrace();

                            Toast toast =
                                    Toast.makeText(
                                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                            toast.show();
                            finish();
                        }
                    }
                    break;
                    default: {
                        super.onManagerConnected(status);} break;
                }
            }
            catch (Exception e){}
        }
    };

    private void showAddFaceDialog(final SimilarityClassifier.Recognition rec) {

        final AlertDialog.Builder builder = new AlertDialog.Builder(this);
        LayoutInflater inflater = getLayoutInflater();
        View dialogLayout = inflater.inflate(R.layout.add_image_dialog2, null);
        ImageView ivFace = dialogLayout.findViewById(R.id.dlg_image);
        final EditText etName = dialogLayout.findViewById(R.id.dlg_input);
        ivFace.setImageBitmap(rec.getCrop());
        etName.setHint("Input name");
        Button conferma=dialogLayout.findViewById(R.id.confermaImg);
        conferma.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String name = etName.getText().toString();
                rec.setTitle(name);
                if (name.isEmpty()) {
                    Toast toast=Toast.makeText(
                            getApplicationContext(), "Inserire un nome nella zona input", Toast.LENGTH_LONG);
                    toast.show();

                    return;
                }
                List<SimilarityClassifier.Recognition>uu=detector.recognizeImage(rec.getCrop(),true);
                if(uu.size()>0){rec.setExtra(uu.get(0).getExtra());detector.register(name, rec);Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Registrato "+rec.getTitle(), Toast.LENGTH_LONG);
                    toast.show();

                }
                System.out.println(rec.getExtra());




            }
        });
        Button confronta=dialogLayout.findViewById(R.id.confrontaImg);
        confronta.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //System.out.println(detector.getDataset().get("pino").getExtra());
               List<SimilarityClassifier.Recognition>results=detector.recognizeImage(rec.getCrop(),false);

                SimilarityClassifier.Recognition r=results.get(0);

                    Toast toast =
                            Toast.makeText(
                                    getApplicationContext(), "Rilevato " + r.getTitle() + " con distanza: " + r.getDistance(), Toast.LENGTH_LONG);
                    toast.show();





            }
        });


        builder.setView(dialogLayout);
        builder.show();

    }





}
