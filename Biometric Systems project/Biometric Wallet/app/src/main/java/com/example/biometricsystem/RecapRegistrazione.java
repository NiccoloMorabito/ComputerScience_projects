package com.example.biometricsystem;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.biometricsystem.controller.IotaService;
import com.example.biometricsystem.model.IotaGenericResponse;

import org.json.JSONArray;
import org.json.JSONObject;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Arrays;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class RecapRegistrazione extends AppCompatActivity {
    private DatasetParser dataset;
    private IotaService iotaService;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recap_registrazione);
        final byte[] byteArray = getIntent().getByteArrayExtra("gray");
        Bitmap bmp = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
        final byte[] byteArray2 = getIntent().getByteArrayExtra("rgba");
        Bitmap bmp2 = BitmapFactory.decodeByteArray(byteArray2, 0, byteArray2.length);
        final String nome=(String) getIntent().getStringExtra("nome");
        final String pw=(String) getIntent().getStringExtra("password");
        final String Seed=(String) getIntent().getStringExtra("seed");
        final String azureProfileId=(String) getIntent().getStringExtra("azureProfileId");
        this.iotaService= new IotaService();
        ((TextView)findViewById(R.id.nomeCompleta)).setText(nome);
        ((TextView)findViewById(R.id.passwordCompleta)).setText(pw);
        ((ImageView)findViewById(R.id.img1)).setImageBitmap(bmp);
        ((ImageView)findViewById(R.id.img2)).setImageBitmap(bmp2);
        float[][] arrayReceived=null;
        dataset=new DatasetParser();
        Object[] objectArray = (Object[]) getIntent().getExtras().getSerializable("extra");
        if(objectArray!=null){
            arrayReceived = new float[objectArray.length][];
            for(int i=0;i<objectArray.length;i++){
                arrayReceived[i]=(float[]) objectArray[i];
            }
        }
        final float[][] arrayRec2=arrayReceived;
        final String id=getIntent().getStringExtra("id");
        final float distance=getIntent().getFloatExtra("distance",-69);
        final Integer color=(Integer) getIntent().getSerializableExtra("color");
        final float top=getIntent().getFloatExtra("top",-69);
        final float bottom=getIntent().getFloatExtra("bottom",-69);
        final float right=getIntent().getFloatExtra("right",-69);
        final float left=getIntent().getFloatExtra("left",-69);
        System.out.println(id+" "+nome+" "+distance+" "+color+" "+top+" "+bottom+" "+right+" "+left);
        findViewById(R.id.completaReg).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    String hash=md5(Arrays.toString(arrayRec2[0]));
                    JSONObject utente=dataset.parseUtente(nome, id, distance, color, top,
                            bottom, right, left, byteArray2, byteArray,
                            pw, arrayRec2,Seed, azureProfileId, hash);
                    dataset.insertDataset(utente,getBaseContext());
                    JSONArray array=dataset.readDataset(getBaseContext());
                    System.out.println("ok");

                    createIotaWallet(azureProfileId,hash);
                    finish();
                }
                catch (Exception e){System.out.println(e.toString());}
            }
        });
    }


    public String md5(String s) {
        try {
            // Create MD5 Hash
            MessageDigest digest = java.security.MessageDigest.getInstance("MD5");
            digest.update(s.getBytes());
            byte messageDigest[] = digest.digest();

            // Create Hex String
            StringBuffer hexString = new StringBuffer();
            for (int i=0; i<messageDigest.length; i++)
                hexString.append(Integer.toHexString(0xFF & messageDigest[i]));
            return hexString.toString();

        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return "";
    }


    private void createIotaWallet(String name, String pass){
        Callback callback=new Callback<IotaGenericResponse>() {
            @Override
            public void onResponse(Call<IotaGenericResponse> call, Response<IotaGenericResponse> response) {

                Log.d("TAG",response.code()+"");
                if(response.isSuccessful()) {
                    IotaGenericResponse resource = response.body();
                    if(resource.success) {
                        Toast.makeText(getApplicationContext(), "Iota Wallet succesfully done!", Toast.LENGTH_SHORT).show();
                        finish();
                    } else {
                        //progress.setVisibility(View.GONE);
                    }
                }

            }

            @Override
            public void onFailure(Call<IotaGenericResponse> call, Throwable t) {
                call.cancel();
                Toast.makeText(getApplicationContext(), "Error with IOTA call, please retry.", Toast.LENGTH_SHORT).show();
                Log.e("IotaWallet", "Error creating IOTA wallet", t);
            }
        };

        this.iotaService.create(name,pass,callback);
    }
    
    }



