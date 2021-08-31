package com.example.biometricsystem;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.Trace;
import android.text.TextUtils;
import android.util.Pair;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter.Options;
import org.tensorflow.lite.Interpreter;



public class TFLiteObjectDetectionAPIModel implements SimilarityClassifier {


    //private static final int OUTPUT_SIZE = 512;
    private static final int OUTPUT_SIZE = 192;

    // Only return this many results.
    private static final int NUM_DETECTIONS = 1;

    // Float model
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;

    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    private boolean isModelQuantized;
    // Config values.
    private int inputSize;
    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private float[][][] outputLocations;
    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] outputClasses;
    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private float[][] outputScores;
    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private float[] numDetections;

    private float[][] embeedings;
    private ByteBuffer imgData;

    private Interpreter tfLite;
    private Interpreter.Options tfLiteOptions;

    // Face Mask Detector Output
    private float[][] output;
    /**Registrazione per l'enrolment**/
    private HashMap<String, Recognition> registered = new HashMap<>();
    public void register(String name, Recognition rec) {
        registered.put(name, rec);
    }

    private TFLiteObjectDetectionAPIModel() {}

    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    /**
     * Inizializzazione oggetto per la classficazione delle immagini
     *
     * @param assetManager asset manager utilizzato per caricare gli asset
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename Il percorso del file delle label delle classi
     * @param inputSize La dimesione dell'immagine di input
     * @param isQuantized Boolean representing model is quantized or not
     */
    public static SimilarityClassifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputSize,
            final boolean isQuantized)
            throws IOException  {
        /**
         * Istanziazione dell'oggetto con caricamento del modello ed impostazione dei parametri
         * **/
        final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {

            d.labels.add(line);
        }
        br.close();

        d.inputSize = inputSize;

        try {
            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.inputSize * d.inputSize];
        d.tfLite.setNumThreads(NUM_THREADS);
        d.outputLocations = new float[1][NUM_DETECTIONS][4];
        d.outputClasses = new float[1][NUM_DETECTIONS];
        d.outputScores = new float[1][NUM_DETECTIONS];
        d.numDetections = new float[1];
        return d;
    }
    
    public HashMap<String,Recognition> getDataset(){return registered;}

    // looks for the nearest embeeding in the dataset (using L2 norm)
    // and retrurns the pair <id, distance>
    /**
     * Controlla l'embeeding nel dataset usando la normalizzazione L2 e restituisce coppie<id,distanza>
     * **/
    private Pair<String, Float> findNearest(float[] emb) {

        Pair<String, Float> ret = null;
        for (Map.Entry<String, Recognition> entry : registered.entrySet()) {
            final String name = entry.getKey();
            final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

            float distance = 0;
            for (int i = 0; i < emb.length; i++) {
                float diff = emb[i] - knownEmb[i];
                distance += diff*diff;
            }
            distance = (float) Math.sqrt(distance);
            if (ret == null || distance < ret.second) {
                ret = new Pair<>(name, distance);
            }
        }
        return ret;
    }

    private Float findUser(float[] emb,String search){

        Pair<String, Float> ret = null;
        Boolean found=false;
        for (Map.Entry<String, Recognition> entry : registered.entrySet()) {
            found=true;
            final String name = entry.getKey();
            if(name.equals(search)) {
                final float[] knownEmb = ((float[][]) entry.getValue().getExtra())[0];

                float distance = 0;
                for (int i = 0; i < emb.length; i++) {
                    float diff = emb[i] - knownEmb[i];
                    distance += diff * diff;
                }
                distance = (float) Math.sqrt(distance);
                if (ret == null || distance < ret.second) {
                    ret = new Pair<>(name, distance);
                }
            }
        }
        if (found){
        return ret.second;}
        else{return -1f;}
    }

    public float verificate(final Bitmap bitmap,String nome){
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        /**Normalizzazione della bitmap di input e preprocessing**/
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        Trace.endSection();
        Trace.beginSection("feed");
        Object[] inputArray = {imgData};
        Trace.endSection();
        /**output map per il Face Mask Detector**/
        Map<Integer, Object> outputMap = new HashMap<>();
        embeedings = new float[1][OUTPUT_SIZE];
        outputMap.put(0, embeedings);
        /**Inference call-----------------------------------------------------------**/
        Trace.beginSection("run");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();
        /**Calcolo del embeeding con distanza più vicina**/
        float distance = Float.MAX_VALUE;
        if (registered.size() > 0) {
            final float nearest = findUser(embeedings[0],nome);
            if (nearest != -1f) {
                distance = nearest;
            }
            else{distance=-1f;}
        }
        Trace.endSection();
        return distance;
    }

    /**
     *
     * Metodo per il Riconoscimento facciale
     */
    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap, boolean storeExtra) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        /**Normalizzazione della bitmap di input e preprocessing**/
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        Trace.endSection();
        Trace.beginSection("feed");
        Object[] inputArray = {imgData};
        Trace.endSection();
        /**output map per il Face Mask Detector**/
        Map<Integer, Object> outputMap = new HashMap<>();
        embeedings = new float[1][OUTPUT_SIZE];
        outputMap.put(0, embeedings);
        /**Inference call-----------------------------------------------------------**/
        Trace.beginSection("run");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();
        /**Calcolo del embeeding con distanza più vicina**/
        float distance = Float.MAX_VALUE;
        String id = "0";
        String label = "?";
        if (registered.size() > 0) {
            final Pair<String, Float> nearest = findNearest(embeedings[0]);
            if (nearest != null) {

                final String name = nearest.first;
                label = name;
                distance = nearest.second;
                //System.out.println("nearest: " + name + " - distance: " + distance);
            }

        }

        /**Instanzazione dell'oggetto di output recognition**/
        final int numDetectionsOutput = 1;
        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);
        Recognition rec = new Recognition(
                id,
                label,
                distance,
                new RectF());

        recognitions.add( rec );

        if (storeExtra) {
            rec.setExtra(embeedings);
        }

        Trace.endSection();
        return recognitions;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {}

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {}

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLite != null) tfLiteOptions.setUseNNAPI(isChecked);
    }

    public HashMap<String, Recognition> loadDataset(Context context) throws IOException, JSONException {


        DatasetParser ds=new DatasetParser();
        JSONArray array=ds.readDataset(context);
        int i=0;
        while(i<array.length()){
            JSONObject utente=(JSONObject) array.get(i);
            String nome=(String) utente.get("name");
            String password=(String) utente.get("password");
            String id=(String) utente.get("id");
            float distance=BigDecimal.valueOf(utente.getDouble("distance")).floatValue();
            float top=BigDecimal.valueOf(utente.getDouble("top")).floatValue();
            float bottom=BigDecimal.valueOf(utente.getDouble("bottom")).floatValue();
            float left=BigDecimal.valueOf(utente.getDouble("let")).floatValue();
            float right=BigDecimal.valueOf(utente.getDouble("right")).floatValue();
            JSONArray rgbaJ=utente.getJSONArray("rgba");
            JSONArray grayJ=utente.getJSONArray("grey");
            byte[] rgba=rgbaJ.toString().getBytes();
            byte[] gray=grayJ.toString().getBytes();
            String seed=utente.getString("seed");


            Object extraJ_righe=utente.get("extra");
            float[][] extra=parseExtra(extraJ_righe.toString());


            SimilarityClassifier.Recognition recognition=new SimilarityClassifier.Recognition(id,nome,distance,new RectF(left,top,right,bottom));
            recognition.setExtra(extra);
            recognition.setRgba(BitmapFactory.decodeByteArray(rgba, 0, rgba.length));
            recognition.setCrop(BitmapFactory.decodeByteArray(gray, 0, gray.length));
            register(recognition.getTitle(),recognition);

            i++;
        }
        return registered;

    }

    public float[][] parseExtra(String string){
        String floatStr = string.substring(1, string.length()-1);
        String[] valuesArr = floatStr.split("],");
        float[][] floatArr = new float[valuesArr.length][];

        for (int i = 0; i < valuesArr.length; i++) {

            String stringaRiga = valuesArr[i].replace("[", "").replace("]", "").trim();

            String[] valuesArr2=stringaRiga.split(",");

           float[] floatArr2=new float[valuesArr2.length];
            for(int j=0;j<valuesArr2.length;j++){

                if (TextUtils.isEmpty(stringaRiga) || TextUtils.isEmpty(stringaRiga.trim())) {
                    floatArr2[j] = 0.0f;
                    continue;
                }
                floatArr2[j] = Float.parseFloat(valuesArr2[j]);
            }
            floatArr[i]=floatArr2;

        }
        return floatArr;


    }


}
