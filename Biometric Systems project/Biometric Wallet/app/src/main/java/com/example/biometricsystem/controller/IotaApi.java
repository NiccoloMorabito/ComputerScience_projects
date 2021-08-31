package com.example.biometricsystem.controller;

import com.example.biometricsystem.model.AzureEnrollResponse;
import com.example.biometricsystem.model.IotaGenericResponse;
import com.example.biometricsystem.model.IotaListAddressResponse;


import java.util.List;

import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.http.Body;
import retrofit2.http.POST;
import retrofit2.http.Path;

public interface IotaApi {

    String CREATE = "createAccount";
    String TRANSFER="Transfer";
    String ADDRESS="ListAddress";

    @POST(CREATE)
    Call<IotaGenericResponse> create(@Body RequestBody body);

    @POST(TRANSFER)
    Call<IotaGenericResponse> transfer(@Body RequestBody body);

    @POST(ADDRESS)
    Call<List<IotaListAddressResponse>> address(@Body RequestBody body);

}
