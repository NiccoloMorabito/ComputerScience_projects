package com.example.biometricsystem.controller;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class IotaClient {

    private static Retrofit retrofit = null;
    private static final String BASE_URL = "https://biometricwallet.herokuapp.com/";

    OkHttpClient client = new OkHttpClient.Builder().callTimeout(1, TimeUnit.MINUTES).build();
    public static Retrofit getClient() {

        retrofit = new Retrofit.Builder()
                .baseUrl(BASE_URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        return retrofit;
    }

}
