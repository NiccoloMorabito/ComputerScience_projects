package com.example.biometricsystem;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AbsListView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.biometricsystem.controller.IotaService;
import com.example.biometricsystem.model.IotaListAddressResponse;

import java.lang.reflect.Array;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;


public class Home extends AppCompatActivity {
    private String name;
    private String hash;
    private String password;
    private String seed;
    private String azureProfileId;
    private ListView listview;
    private List<IotaListAddressResponse> listAddress;
    private IotaService iotaService;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);
        name=getIntent().getStringExtra("nome");
        hash=getIntent().getStringExtra("hash");
        //seed=getIntent().getStringExtra("seed");
        password=getIntent().getStringExtra("password");
        azureProfileId=getIntent().getStringExtra("azureProfileId");
        ((TextView)findViewById(R.id.home_name)).setText(name);
        this.iotaService=new IotaService();
        listview = this.findViewById(R.id.home_Address);
        getAddressList(azureProfileId, hash);
        Button sendmoney = findViewById(R.id.sendMoney);
        sendmoney.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent  intent = new Intent(getBaseContext(), Transaction.class);
                intent.putExtra("azureProfileId", azureProfileId);
                intent.putExtra("hash", hash);
                startActivity(intent);
            }
        });

    }
    @Override
    protected void onResume() {
        super.onResume();
        getAddressList(azureProfileId, hash);
    }

    private void getAddressList(String name, String pass){
        Callback callback=new Callback<List<IotaListAddressResponse>>() {
            @Override
            public void onResponse(Call<List<IotaListAddressResponse>> call, Response<List<IotaListAddressResponse>> response) {

                Log.d("TAG",response.code()+"");

                if(response.isSuccessful()) {
                    List<IotaListAddressResponse> resource = response.body();

                    Log.d("chiamata", String.valueOf(response.body()));
                    Log.d("chiamata", String.valueOf(response.body()));
                    if(!resource.isEmpty())
                    {
                        listAddress=resource;
                        AddressListAdapter adapter = new AddressListAdapter(getApplicationContext(), R.layout.address_adapter, listAddress);
                        listview.setAdapter(adapter);
                    }
                    else
                    {
                        // mTxtEmptyListDispenser.setVisibility(View.VISIBLE);
                        listview.setVisibility(View.GONE);
                    }
                }

            }

            @Override
            public void onFailure(Call<List<IotaListAddressResponse>> call, Throwable t) {
                call.cancel();
                Toast.makeText(getApplicationContext(), "Error with IOTA call, please retry.", Toast.LENGTH_SHORT).show();
                Log.e("IotaWallet", "Error List Address IOTA wallet", t);
            }
        };

        this.iotaService.address(name,pass,callback);
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

    public class AddressListAdapter extends ArrayAdapter<IotaListAddressResponse>  {
        Context mCtx;
        int resource;
        List<IotaListAddressResponse> address_list;

        public AddressListAdapter(Context mCtx , int resource , List<IotaListAddressResponse> address)
        {
            super(mCtx,resource,address);
            this.mCtx = mCtx;
            this.resource = resource;
            this.address_list = address;

        }

        @NonNull
        @Override
        public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
            LayoutInflater inflater = LayoutInflater.from(mCtx);

            View view = inflater.inflate(resource,null);
            TextView textViewName = view.findViewById(R.id.addressElement);
            TextView textViewAmount = view.findViewById(R.id.addressAmount);


            IotaListAddressResponse address = address_list.get(position);
            textViewName.setText(address.address);

            textViewAmount.setText(String.format("%.2f",address.balance));
            Log.d("address", address.address);
            Log.d("amount", String.format("%.2f",address.balance));
            //status.setImageDrawable(mCtx.getResources().getDrawable(dispenser.getImage()));
            // status.setImageResource(0);


            return view;
        }
    }
}