package com.example.biometricsystem.controller;

import com.example.biometricsystem.model.AzureEnrollResponse;
import com.example.biometricsystem.model.AzureIdentifyResponse;
import com.example.biometricsystem.model.AzureRegisterResponse;
import com.example.biometricsystem.model.Locale;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.Headers;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;
import retrofit2.http.Path;

public interface AzureApi {

    String IDENTIFY = "speaker/verification/v2.0/text-independent/profiles/{id}/verify";
    String ENROLL = "speaker/verification/v2.0/text-independent/profiles/{id}/enrollments";
    String REGISTER = "speaker/verification/v2.0/text-independent/profiles";
    static final String API_KEY = "b237ee1bf84b446e803b1e32a2cbfbd9";


    @Headers("Ocp-Apim-Subscription-Key: " + API_KEY)
    @POST(ENROLL)
    Call<AzureEnrollResponse> enroll(@Path("id") String profileId, @Body RequestBody file);

    @Headers("Ocp-Apim-Subscription-Key: " + API_KEY)
    @POST(IDENTIFY)
    Call<AzureIdentifyResponse> identify(@Path("id") String profileId, @Body RequestBody file);

    @Headers("Ocp-Apim-Subscription-Key: " + API_KEY)
    @POST(REGISTER)
    Call<AzureRegisterResponse> register(@Body Locale locale);
}
