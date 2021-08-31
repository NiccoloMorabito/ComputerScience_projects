package com.example.biometricsystem;

import androidx.appcompat.app.AppCompatActivity;

import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.provider.Settings;
import android.util.AttributeSet;
import android.util.Pair;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Timer;

public class Recognition extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    JavaCameraView javaCameraView;
    File cascadeFile;
    CascadeClassifier faceDetector;
    private Mat mRgba,mGrey;
    // MobileFaceNet
    private static final int TF_OD_API_INPUT_SIZE = 112;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";
    private double relativeDistance=0.9;

    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private int cameraIndex;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

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
    public boolean detected;
    private Bitmap faceBmp = null;
    private Bitmap croppedFaceBmp;
    private  Bitmap croppedFaceBmp2;
    android.graphics.Rect faceRect;
    private Button fabAdd;
   private  FaceDetectorOptions realTimeOpts;
    private int recognitionsCounter;
    private String target;
    private String seed;
    private String azureProfileId;
    private String password;
    private int counter=10;
    private String[] azioni={"chiudi occhio destro","chiudi occhio sinistro","chiudi occhi","sorridi"};
    private int[] sequenza=new int[3];
    private boolean duringTest=false;
    private int indiceSequenza=0;
    private Button actual;
    private String hash;

    private void swapCamera() {
        if(cameraIndex==0){cameraIndex=1;}
        else{cameraIndex=0;}
        javaCameraView.disableView();
        javaCameraView.setCameraIndex(cameraIndex);
        javaCameraView.enableView();
    }
    @Override
    public View onCreateView(String name, Context context, AttributeSet attrs)
    {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        faceRect = new android.graphics.Rect();

        return super.onCreateView(name, context, attrs);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognition);
        javaCameraView=(JavaCameraView) findViewById(R.id.javacameraview_rec);
        cameraIndex=0;
        javaCameraView.setCameraIndex(0);
        recognitionsCounter=0;
        target=getIntent().getStringExtra("nome");
        realTimeOpts = new FaceDetectorOptions.Builder()
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .build();
        if(!OpenCVLoader.initDebug())
        {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,callback);
        }
        else{
            callback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        javaCameraView.setCvCameraViewListener(this);
        try {
            System.out.println(detector.getDataset().get("provona"));

        }
        catch(Exception e){System.out.println(e.getCause());}
        Button switchCamera=(Button) findViewById(R.id.switch_camera_rec);
        switchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                swapCamera();
            }
        });
        password=getIntent().getStringExtra("password");
        seed=getIntent().getStringExtra("seed");
        hash=getIntent().getStringExtra("hash");
        azureProfileId=getIntent().getStringExtra("azureProfileId");
        final Button start=(Button) findViewById(R.id.start);
        final Button timer=(Button) findViewById(R.id.timer);
        final Context c=this;
        actual=(Button)findViewById(R.id.actual);
        actual.setVisibility(View.GONE);
        timer.setVisibility(View.GONE);
        start.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v) {
                int w=0;
                indiceSequenza=0;
                duringTest=true;
                String text="";
                while (w<=2){
                   int numero=(int)(Math.random() * (3 - 0 + 1) + 0);
                    if(w!=0){
                        if(sequenza[w-1]!=numero){sequenza[w]=numero;text+=azioni[numero]+" ; ";w++;}
                        else{continue;}
                    }
                    else{
                        sequenza[w]=numero;
                        text+=azioni[numero]+" ; ";
                        w++;
                    }

                }
                actual.setText(azioni[sequenza[indiceSequenza]]);
                actual.setVisibility(View.VISIBLE);
                System.out.println(text);
                start.setVisibility(View.GONE);
                System.out.println(Arrays.toString(sequenza));
                timer.setVisibility(View.VISIBLE);
                new CountDownTimer(10000, 1000){
                    public void onTick(long millisUntilFinished){
                        timer.setText(String.valueOf(counter));
                        counter--;
                    }
                    public  void onFinish(){

                        timer.setText("10");
                        counter=10;
                        indiceSequenza=0;
                        timer.setVisibility(View.GONE);
                        start.setVisibility(View.VISIBLE);
                        actual.setVisibility(View.GONE);
                        duringTest=false;
                    }
                }.start();}});
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
    private void prepareDetector(){

        DatasetParser ds=new DatasetParser();
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
                final float occhioDx=maxFace.getLeftEyeOpenProbability();
                final float occhioSx=maxFace.getRightEyeOpenProbability();
                final float smile=maxFace.getSmilingProbability();
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
                Imgproc.rectangle(mRgba, new Point(r.left, r.top), new Point(r.right, r.bottom), new Scalar(255, 0, 0));
                final TFLiteObjectDetectionAPIModel det=this.detector;
                Runnable task = new Runnable() {
                    @Override
                    public void run() {
                       // List<SimilarityClassifier.Recognition>results=det.recognizeImage(croppedFaceBmp,false);
                        final float dist=det.verificate(croppedFaceBmp,target);
                      //  final SimilarityClassifier.Recognition r=results.get(0);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {

                                int code=livenessDetection(occhioDx,occhioSx,smile);
                                if(code==3||code==1||code==2||code==0){System.out.println(code);}
                                if ((code==sequenza[indiceSequenza])){
                                    System.out.println(code+" "+Arrays.toString(sequenza));
                                    if (duringTest&&(dist<=relativeDistance&&dist!=-1f)) {
                                        if (indiceSequenza == 2) {
                                            indiceSequenza = 0;
                                            Intent intent = new Intent(getBaseContext(), Home.class);
                                            intent.putExtra("nome",target).putExtra("password",password).putExtra("seed",seed).putExtra("azureProfileId", azureProfileId).putExtra("hash",hash);
                                            startActivity(intent);
                                            finish();

                                        } else {
                                            indiceSequenza++;
                                            actual.setText(azioni[sequenza[indiceSequenza]]);
                                        }
                                    }
                                }
                            }
                        });
                    }
                };
                AsyncTask.execute(task);
                detected=true;
            }
            else{
                detected=false;
            }
        }
        catch (Exception e){}

        return mRgba;
    }
    private int livenessDetection(float occhioDx, float occhioSx, float smile){

        if((occhioDx<=0.4 )&& (occhioSx>=0.6)&&(smile<=0.4)){return 0;}
        if((occhioDx>=0.6 )&& (occhioSx<=0.4)&&(smile<0.4)){return 1;}
        if((occhioDx<=0.3 )&& (occhioSx<=0.3)&&(smile<=0.4)){return 2;}
        if((occhioDx>=0.5 )&& (occhioSx>=0.5)&&(smile>=0.6)){return 3;}
        else{return 4;}


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
                           detector.loadDataset(getBaseContext());
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

@Override
public boolean onKeyDown(int keyCode, KeyEvent event)
{
    if ((keyCode == KeyEvent.KEYCODE_BACK))
    {
        finish();
    }
    return super.onKeyDown(keyCode, event);
}

}
