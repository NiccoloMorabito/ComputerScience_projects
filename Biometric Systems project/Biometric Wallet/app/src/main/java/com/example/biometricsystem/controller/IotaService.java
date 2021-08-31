package com.example.biometricsystem.controller;

import android.util.ArrayMap;

import com.example.biometricsystem.model.AzureEnrollResponse;
import com.example.biometricsystem.model.AzureIdentifyResponse;
import com.example.biometricsystem.model.AzureRegisterResponse;
import com.example.biometricsystem.model.IotaGenericResponse;
import com.example.biometricsystem.model.IotaListAddressResponse;
import com.example.biometricsystem.model.Locale;

import java.io.File;
import java.util.List;
import java.util.Map;

import androidx.annotation.NonNull;

import org.json.JSONObject;

import okhttp3.MediaType;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;

public class IotaService {

    IotaApi iotaApi;

    public IotaService() {
        iotaApi = IotaClient.getClient().create(IotaApi.class);
    }

    public void create(String name, String pass, Callback callback) {
        Map<String, Object> jsonParams = new ArrayMap<>();
        //put something inside the map, could be null
        jsonParams.put("name", name);
        jsonParams.put("pass", pass);
        RequestBody body=RequestBody.create(okhttp3.MediaType.parse("application/json; charset=utf-8"),(new JSONObject(jsonParams)).toString());
        Call<IotaGenericResponse> call = this.iotaApi.create(body);
        call.enqueue(callback);
    }

    public void address(String name, String pass, Callback callback) {
        Map<String, Object> jsonParams = new ArrayMap<>();
        //put something inside the map, could be null
        jsonParams.put("name", name);
        jsonParams.put("pass", pass);
        RequestBody body=RequestBody.create(okhttp3.MediaType.parse("application/json; charset=utf-8"),(new JSONObject(jsonParams)).toString());
        Call<List<IotaListAddressResponse>> call = this.iotaApi.address(body);
        call.enqueue(callback);
    }

    public void transfer(String name, String pass, Double amount, String address, Callback callback) {
        Map<String, Object> jsonParams = new ArrayMap<>();
        //put something inside the map, could be null
        jsonParams.put("name", name);
        jsonParams.put("pass", pass);
        jsonParams.put("amount", amount);
        jsonParams.put("address", address);
        RequestBody body=RequestBody.create(okhttp3.MediaType.parse("application/json; charset=utf-8"),(new JSONObject(jsonParams)).toString());
        Call<IotaGenericResponse> call = this.iotaApi.transfer(body);
        call.enqueue(callback);
    }



}
