package com.example.biometricsystem;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.example.biometricsystem.controller.IotaApi;
import com.example.biometricsystem.controller.IotaService;
import com.example.biometricsystem.controller.RecognitoService;
import com.example.biometricsystem.model.AzureIdentifyResponse;
import com.example.biometricsystem.model.AzureRegisterResponse;
import com.example.biometricsystem.model.IotaGenericResponse;
import com.example.biometricsystem.model.IotaListAddressResponse;
import com.example.biometricsystem.model.ServerResponse;
import com.google.android.material.textfield.TextInputEditText;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;

import cafe.adriel.androidaudioconverter.AndroidAudioConverter;
import cafe.adriel.androidaudioconverter.callback.IConvertCallback;
import cafe.adriel.androidaudioconverter.callback.ILoadCallback;
import cafe.adriel.androidaudioconverter.model.AudioFormat;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class Transaction extends AppCompatActivity {

    private static final String LOG_TAG = "AudioRecordTest";
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static List<String> fileNames = null;
    private Transaction.RecordButton recordButton = null;
    private MediaRecorder recorder = null;
    private Transaction.ConvertButton convertButton = null;
    private MediaPlayer player = null;
    // Requesting permission to RECORD_AUDIO
    private boolean permissionToRecordAccepted = false;
    private String[] permissions = {Manifest.permission.RECORD_AUDIO};
    private List <File> convertedFilesList = new ArrayList();
    private static final int AUDIO_NUM = 1;
    private String Seed;
    RecognitoService recognitoService;
    private TextInputEditText amountTextEdit;
    private TextInputEditText addressTextEdit;
    private ProgressBar progress;
    private String azureProfileId;
    private String hash;
    private IotaService iotaService;
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_RECORD_AUDIO_PERMISSION:
                permissionToRecordAccepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                break;
        }
        if (!permissionToRecordAccepted) finish();
    }

    @SuppressLint("WrongConstant")
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.recognitoService = new RecognitoService();
        this.fileNames = new ArrayList<String>();
        // Record to the external cache directory for visibility
        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);
        this.iotaService=new IotaService();
        LinearLayout ll = new LinearLayout(this);
        ll.setOrientation(1);
        recordButton = new Transaction.RecordButton(this);
        amountTextEdit= new TextInputEditText(this);
        amountTextEdit.setText("10000000");
        addressTextEdit= new TextInputEditText(this);
        addressTextEdit.setText("atoi1qq82pesy4yaxm385ehr4jll9vy63hl0zkt32vd66tv57j8ydcvx47yqruta");
        progress = new ProgressBar(this);
        progress.setVisibility(View.GONE);
        TextView t1=new TextView(this);
        t1.setText("Insert the value of the transaction");
        ll.addView(t1);
        ll.addView(amountTextEdit,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));

        TextView t2=new TextView(this);
        t2.setText("Insert the address destination of the transaction");
        ll.addView(t2);

        ll.addView(addressTextEdit,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));
        TextView t3=new TextView(this);
        t3.setText("Press 'start recording', record a sentence, stop the recording and press 'send transaction'");
        ll.addView(t3);
        ll.addView(recordButton,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));
        convertButton = new Transaction.ConvertButton(this);
        convertButton.setText("Send transaction");
        ll.addView(convertButton,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));
        ll.addView(progress,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));
        setContentView(ll);

        AndroidAudioConverter.load(this, new ILoadCallback() {
            @Override
            public void onSuccess() {
                Log.d("debug", "FFmpeg supported");
            }

            @Override
            public void onFailure(Exception error) {
                Log.d("debug", "FFmpeg NOT supported");
            }
        });
    }

    @Override
    public void onStop() {
        super.onStop();
        if (recorder != null) {
            recorder.release();
            recorder = null;
        }

        if (player != null) {
            player.release();
            player = null;
        }
    }

    private void onRecord(boolean start, int idx) {
        if (start) {
            startRecording(idx);
        } else {
            stopRecording(idx);
        }
    }

    private void onConvert(boolean start) throws FileNotFoundException {
        if (start) {
            startConverting();
        }
    }

    private void startConverting() throws Error {
        progress.setVisibility(View.VISIBLE);
        convertButton.setVisibility(View.GONE);

        azureProfileId = getIntent().getStringExtra("azureProfileId");

        this.recognitoService.identify(azureProfileId, convertedFilesList.get(0), new Callback<AzureIdentifyResponse>() {
            @Override
            public void onResponse(Call<AzureIdentifyResponse> call, Response<AzureIdentifyResponse> response) {

                Log.d("TAG",response.code()+"");

                if(response.isSuccessful()) {
                    AzureIdentifyResponse resource = response.body();
                    if(resource.recognitionResult.equals("Accept")) {

                        //TODO: Iota send transaction api
                        /*
                        https://biometricwallet.herokuapp.com/Transfer
                        Input (body):name
                                     pass
                                     amount
                                     address
                        Output: ok
                                error
                         */
                        //azureProfileId=getIntent().getStringExtra("azureProfileId");
                        Toast.makeText(getApplicationContext(), "USER VERIFIED! Creating new transaction...", Toast.LENGTH_LONG).show();
                        hash=getIntent().getStringExtra("hash");
                        Log.d("blabla",amountTextEdit.getText().toString());
                        Log.d("car",addressTextEdit.getText().toString());
                        transaction(azureProfileId,hash,amountTextEdit.getText().toString(),addressTextEdit.getText().toString());
                    } else {
                        Log.d("Transaction", "USER NOT VERIFIED");
                        Toast.makeText(getApplicationContext(), "USER NOT VERIFIED!", Toast.LENGTH_SHORT).show();
                        finish();
                    }
                }
                progress.setVisibility(View.GONE);
                convertButton.setVisibility(View.VISIBLE);
            }

            @Override
            public void onFailure(Call<AzureIdentifyResponse> call, Throwable t) {
                call.cancel();
                Toast.makeText(getApplicationContext(), "Error with Azure call, please retry.", Toast.LENGTH_SHORT).show();
                Log.e("VocalTransaction", "Error sending vocal registration to Azure server", t);
            }

        });

    }


    private void transaction(String name, String pass, String amount, String address){
        Callback callback=new Callback<IotaGenericResponse>() {
            @Override
            public void onResponse(Call<IotaGenericResponse> call, Response<IotaGenericResponse> response) {

                Log.d("TAG",response.code()+"");

                if(response.isSuccessful()) {
                    IotaGenericResponse resource = response.body();

                    if(resource.success) {


                        Toast.makeText(getApplicationContext(), "Transaction succesfully done!", Toast.LENGTH_SHORT).show();


                        finish();
                    } else {

                        //progress.setVisibility(View.GONE);

                    }
                }

            }

            @Override
            public void onFailure(Call<IotaGenericResponse> call, Throwable t) {
                call.cancel();
                Toast.makeText(getApplicationContext(), "Error with IOTA network, please retry.", Toast.LENGTH_SHORT).show();
                Log.e("IotaWallet", "Error Transaction IOTA", t);
            }
        };

        Double am = Double.parseDouble(amount);
        this.iotaService.transfer(name,pass,am,address,callback);
    }


    private void startRecording(int idx) {
        recorder = new MediaRecorder();
        fileNames.add(new StringBuilder()
                .append(getExternalCacheDir().getAbsolutePath())
                .append("/audiorecordtestone_")
                .append(idx)
                .append(".m4a")
                .toString());
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        recorder.setOutputFile(fileNames.get(idx));
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        recorder.setAudioEncodingBitRate(16*48000);
        recorder.setAudioSamplingRate(48000);
        try {
            recorder.prepare();
        } catch (IOException e) {
            Log.e(LOG_TAG, "prepare() failed");
        }

        recorder.start();
    }

    private void stopRecording(int idx) {
        recorder.stop();
        recorder.release();
        recorder = null;

        IConvertCallback callback = new IConvertCallback() {
            @Override
            public void onSuccess(File convertedFile) {
                Log.d("converter", "Audio convertito con successo");
                convertedFilesList.add(convertedFile);
            }

            @Override
            public void onFailure(Exception error) {
                Log.e("converter", "Errore nella conversione dell'audio");
                throw new Error("Coversion error");
            }
        };

        File f = new File(fileNames.get(idx));

        AndroidAudioConverter.with(this)
                // Your current audio file
                .setFile(f)

                // Your desired audio format
                .setFormat(AudioFormat.WAV)

                // An callback to know when conversion is finished
                .setCallback(callback)

                // Start conversion
                .convert();
    }

    private void convertAudioToWav(File file, IConvertCallback callback) {

        AndroidAudioConverter.with(this)
                // Your current audio file
                .setFile(file)

                // Your desired audio format
                .setFormat(AudioFormat.WAV)

                // An callback to know when conversion is finished
                .setCallback(callback)

                // Start conversion
                .convert();
    }

    class RecordButton extends androidx.appcompat.widget.AppCompatButton {
        boolean mStartRecording = true;
        int idx = 0;

        OnClickListener clicker = new OnClickListener() {
            public void onClick(View v) {
                onRecord(mStartRecording, idx);
                if (mStartRecording) {
                    setText("Stop recording");
                } else {
                    if(++idx < AUDIO_NUM) {
                        setText("Start recording");
                    } else {
                        setText("All recorded");
                        setEnabled(false);
                    }
                }
                mStartRecording = !mStartRecording;
            }
        };

        public RecordButton(Context ctx) {
            super(ctx);
            setText("Start recording");
            setOnClickListener(clicker);
        }
    }

    class ConvertButton extends androidx.appcompat.widget.AppCompatButton {
        boolean mStartConverting = true;

        OnClickListener clicker = new OnClickListener() {
            public void onClick(View v) {
                try {
                    if(fileNames.size() >= AUDIO_NUM) {
                        onConvert(mStartConverting);
                    }
                    else {
                        Toast.makeText(getContext(), String.format("Record %d audio before converting", AUDIO_NUM), Toast.LENGTH_SHORT).show();
                    }
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }
        };

        public ConvertButton(Context ctx) {
            super(ctx);
            setText("Start converting");
            setOnClickListener(clicker);
        }
    }


}
