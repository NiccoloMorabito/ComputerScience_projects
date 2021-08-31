package com.example.biometricsystem.controller;

import com.example.biometricsystem.model.AzureEnrollResponse;
import com.example.biometricsystem.model.AzureIdentifyResponse;
import com.example.biometricsystem.model.AzureRegisterResponse;
import com.example.biometricsystem.model.Locale;

import java.io.File;

import androidx.annotation.NonNull;
import okhttp3.MediaType;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;

public class RecognitoService{

    AzureApi azureApi;

    public RecognitoService() {
        azureApi = AzureClient.getClient().create(AzureApi.class);
    }

    public void enroll(String profileId, File file, Callback callback) {
        Call<AzureEnrollResponse> call = this.azureApi.enroll(profileId, prepareAudioFilePart("audioData", file));
        call.enqueue(callback);
    }

    public void identify(String wallet, File file, Callback callback) {
        Call<AzureIdentifyResponse> call = this.azureApi.identify(wallet, prepareAudioFilePart("audioData", file));
        call.enqueue(callback);
    }

    public void register(Callback callback) {
        Call<AzureRegisterResponse> call = this.azureApi.register(new Locale("it-IT"));
        call.enqueue(callback);
    }

    @NonNull
    private RequestBody prepareAudioFilePart(String partName, File file) {

        // create RequestBody instance from file
        RequestBody requestFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        // MultipartBody.Part is used to send also the actual file name
        return requestFile;
    }
}
