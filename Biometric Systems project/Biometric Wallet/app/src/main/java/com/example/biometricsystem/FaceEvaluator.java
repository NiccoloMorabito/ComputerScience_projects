package com.example.biometricsystem;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

public class FaceEvaluator extends AppCompatActivity {

    private static final String TAG = "FaceEvaluator";

    private final String FACE_DATASET_PATH = "our_dataset";
    private final String IMPOSTOR_PATH = "impostors";
    private final String ENROLL_IMAGE_NAME = "enroll.jpg";

    private final int MIN_THRESHOLD_TO_TEST = 1;
    private final int MAX_THRESHOLD_TO_TEST = 15;

    private AssetManager assetManager;
    private RecognitionUtils recognitionUtils;

    private BaseLoaderCallback callback=new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            try {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS: { }
                    break;
                    default: { super.onManagerConnected(status); }
                    break;
                }
            }
            catch (Exception e){}
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        assetManager = this.getAssets();
        recognitionUtils = new RecognitionUtils(this);

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognition_evaluation);
        if(!OpenCVLoader.initDebug())
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,callback);
        else
            callback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        Runnable r=new Runnable() {
            @Override
            public void run() {
                final String[] people;
                try {
                    people = assetManager.list(FACE_DATASET_PATH);
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.e(TAG, "Dataset not found. Process aborted.");
                    return;
                }

                enrollPeople(people);
                evaluatePeopleAndSaveResults(people);
            }
        };
        AsyncTask.execute(r);
        /**
        Runnable prova=new Runnable() {
            @Override
            public void run() {
                try {
                    InputStream personStream = assetManager.open("DatasetRecognition/enroll.jpg");
                    System.out.println(recognitionUtils.enroll("prova", personStream));
                    InputStream personStream2 = assetManager.open("DatasetRecognition/rec.jpg");
                    System.out.println(recognitionUtils.verificate("prova", personStream2));
                }
                catch (Exception e){System.out.println(e.toString());}
            }
        };
       AsyncTask.execute(prova);
         **/
    }

    private void enrollPeople(String[] people) {
        Log.i(TAG, "Enrollment started.");
        for (String person : people) {
            final String enrollImagePath = getEnrollImagePath(person);
            try {
                InputStream personStream = assetManager.open(enrollImagePath);
                recognitionUtils.enroll(person, personStream);
            } catch (IOException e) {
                e.printStackTrace();
                Log.e(
                    TAG,
                    String.format("%s not enrolled. Error in opening the image at the following path: %s ", person, enrollImagePath)
                );
            } catch (NoFaceDetectedException e) {
                e.printStackTrace();
                Log.e(
                    TAG,
                    String.format("%s not enrolled. No face detected in the image at the following path: %s ", person, enrollImagePath)
                );
            }
        }
        Log.i(TAG, "Enrollment finished.");
    }

    private String getEnrollImagePath(String person) {
        return FACE_DATASET_PATH + "/" + person + "/" + ENROLL_IMAGE_NAME;
    }

    private void evaluatePeopleAndSaveResults(String[] people) {
        // threshold -> metrics
        Map<Double, int[]> thresholdToMetrics = new HashMap<>();
        // threshold -> person -> metrics
        Map<Double, Map<String, int[]>> thresholdToPersonToMetrics = new HashMap<>();
        // metrics are saved in a 4-sized array in the following order: GA, FR, FA, GR

        for (int t=MIN_THRESHOLD_TO_TEST; t<=MAX_THRESHOLD_TO_TEST; t++) {
            double threshold = (double)t/10;
            Log.i(TAG, "Evaluation with threshold: " + threshold);
            recognitionUtils.setThreshold(threshold);

            Map<String, int[]> personToMetrics = new HashMap<>();
            int[] finalMetrics = new int[4];

            for (String person : people) {
                int[] partialMetrics = new int[4];

                // genuine attempts
                String personPath = FACE_DATASET_PATH + "/" + person;
                String[] images;
                try {
                    images = assetManager.list(personPath);
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.e(TAG, "[Evaluation] Impossible to load the images in the following directory: " + personPath);
                    continue;
                }

                for (String image : images) {
                    if (image.equals(ENROLL_IMAGE_NAME))
                        continue;
                    final String imagePath = personPath + "/" + image;
                    boolean isVerified;
                    try {
                        isVerified = recognitionUtils.isVerified(person, assetManager.open(imagePath));
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Error in opening the image at the following path: " + imagePath);
                        continue;
                    } catch (NoFaceDetectedException e) {
                        e.printStackTrace();
                        Log.e(TAG, "No face detected in verifying the image at the following path: " + imagePath);
                        continue;
                    }
                    // GA
                    if (isVerified) {
                        partialMetrics[0] += 1;
                        finalMetrics[0] += 1;
                    }
                    // FR
                    else {
                        partialMetrics[1] += 1;
                        finalMetrics[1] += 1;
                    }

                }

                // impostor attempts
                String[] impostorImages;
                try {
                    impostorImages = assetManager.list(IMPOSTOR_PATH);
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.e(TAG, "Impossible to load the images in the following directory: " + personPath);
                    continue;
                }

                for (String impostorImage : impostorImages) {
                    // skip photos of person in the impostors
                    if (impostorImage.equals(person + ".jpg"))
                        continue;

                    String impostorImagePath = IMPOSTOR_PATH + "/" + impostorImage;
                    boolean isImpostorVerified;
                    try {
                        isImpostorVerified = recognitionUtils.isVerified(person, assetManager.open(impostorImagePath));
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Error in opening the image at the following path: " + impostorImagePath);
                        continue;
                    } catch (NoFaceDetectedException e) {
                        e.printStackTrace();
                        Log.e(TAG, "No face detected in verifying the image at the following path: " + impostorImagePath);
                        continue;
                    }
                    // FA
                    if (isImpostorVerified) {
                        partialMetrics[2] += 1;
                        finalMetrics[2] += 1;
                    }
                    // GR
                    else {
                        partialMetrics[3] += 1;
                        finalMetrics[3] += 1;
                    }
                }

                personToMetrics.put(person, partialMetrics);
            }

            thresholdToMetrics.put(threshold, finalMetrics);
            thresholdToPersonToMetrics.put(threshold, personToMetrics);
            Log.i(TAG, "With threshold: " + threshold + ", the finalMetrics are: " + Arrays.toString(finalMetrics));
        }

        // save results into json files
        String thresholdToMetricsFilename = "(ours+imp) t2metrics_from1to15.json";
        String thresholdToPersonToMetricsFilename = "(ours+imp) t2person2metrics_from1to15.json";
        try {
            new ObjectMapper().writeValue(new File(getFilesDir(), thresholdToMetricsFilename), thresholdToMetrics);
            Log.i(TAG, String.format("Saved the %s file into the %s folder.", thresholdToMetricsFilename, getFilesDir().toString()));
            new ObjectMapper().writeValue(new File(getFilesDir(), thresholdToPersonToMetricsFilename), thresholdToPersonToMetrics);
            Log.i(TAG, String.format("Saved the %s file into the %s folder.", thresholdToPersonToMetricsFilename, getFilesDir().toString()));
        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Error in saving results of evaluation into the " + getFilesDir().toString() + " folder.");
        }
    }

}