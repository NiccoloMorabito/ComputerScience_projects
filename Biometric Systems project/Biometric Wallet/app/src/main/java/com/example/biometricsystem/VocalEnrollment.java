package com.example.biometricsystem;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;


import com.example.biometricsystem.controller.IotaService;
import com.example.biometricsystem.controller.RecognitoService;
import com.example.biometricsystem.model.AzureEnrollResponse;
import com.example.biometricsystem.model.AzureRegisterResponse;
import com.example.biometricsystem.model.IotaGenericResponse;


import java.io.File;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import cafe.adriel.androidaudioconverter.AndroidAudioConverter;
import cafe.adriel.androidaudioconverter.callback.IConvertCallback;
import cafe.adriel.androidaudioconverter.callback.ILoadCallback;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;


public class VocalEnrollment extends AppCompatActivity {


    private static final String LOG_TAG = "AudioRecordTest";
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static String fileName = null;

    private RecordButton recordButton = null;
    private MediaRecorder recorder = null;

    private MediaPlayer player = null;

    // Requesting permission to RECORD_AUDIO
    private boolean permissionToRecordAccepted = false;
    private String[] permissions = {Manifest.permission.RECORD_AUDIO};

    private List <File> convertedFilesList = new ArrayList();

    private static final int AUDIO_NUM = 5;

    private String Seed;
    private String profileId;

    RecognitoService recognitoService;
    private ProgressBar progress;


    Thread recordingThread;
    boolean isRecording = false;
    int audioSource = MediaRecorder.AudioSource.MIC;
    int sampleRateInHz = 16000;
    int channelConfig = AudioFormat.CHANNEL_IN_MONO;
    int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
    int bufferSizeInBytes = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat);
    byte Data[] = new byte[bufferSizeInBytes];
    AudioRecord audioRecorder = new AudioRecord(audioSource,
            sampleRateInHz,
            channelConfig,
            audioFormat,
            bufferSizeInBytes);
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

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.recognitoService = new RecognitoService();
        this.iotaService=new IotaService();
        this.fileName = "";
        this.recognitoService.register(new Callback<AzureRegisterResponse>() {
            @Override
            public void onResponse(Call<AzureRegisterResponse> call, Response<AzureRegisterResponse> response) {
                Log.d("TAG",response.code()+"");
                if(response.isSuccessful()) {
                    AzureRegisterResponse resource = response.body();
                    profileId = resource.profileId;

                    progress.setVisibility(View.GONE);
                    recordButton.setVisibility(View.VISIBLE);
                }
            }
            @Override
            public void onFailure(Call<AzureRegisterResponse> call, Throwable t) {
                call.cancel();
                Toast.makeText(getApplicationContext(), "Error with Azure profile creation", Toast.LENGTH_SHORT).show();
                Log.e("VocalEnrollment", "Error with Azure profile creation.", t);
            }
        });
        // Record to the external cache directory for visibility
        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);
        LinearLayout ll = new LinearLayout(this);
        ll.setOrientation(LinearLayout.VERTICAL);
        recordButton = new RecordButton(this);
        progress = new ProgressBar(this);
        TextView t=new TextView(this);
        t.setText("Please create an audio of 20 seconds. You can stop the audio, but you need to speak for a total time of 20 seconds.");
        recordButton.setVisibility(View.GONE);
        progress.setVisibility(View.VISIBLE);
        ll.addView(recordButton,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));
        ll.addView(progress,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        0));
        ll.addView(t);
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

    private void onRecord(boolean start) {
        if (start) {
            startRecording();
        } else {
            stopRecording();
        }
    }


    private void startRecording() {

        recorder = new MediaRecorder();
        fileName = new StringBuilder()
                .append(getExternalCacheDir().getAbsolutePath())
                .append("/audiorecordtestone.m4a")
                .toString();
        recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        recorder.setOutputFile(fileName);
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

    private void stopRecording() {
        recorder.stop();
        recorder.release();
        recorder = null;

        IConvertCallback callback = new IConvertCallback() {
            @Override
            public void onSuccess(File convertedFile) {
                Log.d("converter", "Audio convertito con successo");
                sendAudioForEnrollment(convertedFile);
                //convertedFilesList.add(convertedFile);
            }

            @Override
            public void onFailure(Exception error) {
                Log.e("converter", "Errore nella conversione dell'audio");
                throw new Error("Coversion error");
            }
        };

        File f = new File(fileName);

        AndroidAudioConverter.with(this)
                // Your current audio file
                .setFile(f)

                // Your desired audio format
                .setFormat(cafe.adriel.androidaudioconverter.model.AudioFormat.WAV)

                // An callback to know when conversion is finished
                .setCallback(callback)

                // Start conversion
                .convert();
    }

    private void sendAudioForEnrollment(File file) {
        progress.setVisibility(View.VISIBLE);
        recordButton.setVisibility(View.GONE);

        this.recognitoService.enroll(profileId, file, new Callback<AzureEnrollResponse>() {
            @Override
            public void onResponse(Call<AzureEnrollResponse> call, Response<AzureEnrollResponse> response) {

                Log.d("TAG",response.code()+"");

                if(response.isSuccessful()) {
                    AzureEnrollResponse resource = response.body();

                    if(resource.remainingEnrollmentsSpeechLength <= 0) {
                        //TODO: IOTA api for seed generation
                        //createIotaWallet(profileId, getIntent().get );
                        RandomString rs = new RandomString(81, new SecureRandom(),"9ABCDEFGHIJKLMNOPQRSTUVWXYZ");
                        Seed = rs.nextString();

                        Intent intent = new Intent(getBaseContext(), RecapRegistrazione.class);
                        intent.putExtras(getIntent().getExtras());
                        intent.putExtra("seed", Seed);
                        intent.putExtra("azureProfileId", profileId);

                        Toast.makeText(getApplicationContext(), "Vocal enrollment succesfully done!", Toast.LENGTH_SHORT).show();

                        startActivity(intent);
                        finish();
                    } else {
                        progress.setVisibility(View.GONE);
                        recordButton.setVisibility(View.VISIBLE);
                    }
                }

            }

            @Override
            public void onFailure(Call<AzureEnrollResponse> call, Throwable t) {
                call.cancel();
                Toast.makeText(getApplicationContext(), "Error with Azure call, please retry.", Toast.LENGTH_SHORT).show();
                Log.e("VocalEnrollment", "Error sending vocal registration to Azure server", t);
            }

        });
    }



    class RecordButton extends androidx.appcompat.widget.AppCompatButton {
        boolean mStartRecording = true;

        OnClickListener clicker = new OnClickListener() {
            public void onClick(View v) {
                onRecord(mStartRecording);
                if (mStartRecording) {
                    setText("Stop recording");
                } else {
                    setText("Start recording");
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

}