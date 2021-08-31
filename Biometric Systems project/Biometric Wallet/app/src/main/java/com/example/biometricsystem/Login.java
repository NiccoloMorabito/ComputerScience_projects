package com.example.biometricsystem;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Login extends AppCompatActivity {
    DatasetParser datasetParser ;
    private static final int TF_OD_API_INPUT_SIZE = 112;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_registrazione);
        ((TextView)findViewById(R.id.reg_titolo)).setText("Login");
        Button go=(Button) findViewById(R.id.go_enroll);
        go.setText("Login");
        datasetParser=new DatasetParser();
        go.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                String  nome=((EditText)findViewById(R.id.registrazione_nome)).getText().toString();
                String password=((EditText)findViewById(R.id.registrazione_password)).getText().toString();
                System.out.println("nome:"+nome+";");
                System.out.println("password:"+password+";");
                if(!nome.equals("")&&!password.equals("")) {
                    Intent intent = new Intent(getBaseContext(), Recognition.class);
                    intent.putExtra("nome", nome).putExtra("password", password);
                    Map<String, String> result = null;
                    try {
                        result = verificateData(nome, password);
                    if (result.size() == 3) {

                        Toast toast =
                                Toast.makeText(getApplicationContext(), "Riconoscimento facciale " + nome, Toast.LENGTH_SHORT);
                        intent.putExtra("seed", result.get("seed"));
                        intent.putExtra("azureProfileId", result.get("azureProfileId"));
                        intent.putExtra("hash", result.get("hash"));
                        startActivity(intent);
                        finish();
                    } else {
                        Toast toast =
                                Toast.makeText(
                                        getApplicationContext(), "Dati inseriti non corretti", Toast.LENGTH_SHORT);
                        toast.show();
                    }
                }catch(Exception e){ Toast toast =
                            Toast.makeText(
                                    getApplicationContext(), "Nessun utente registrato", Toast.LENGTH_SHORT);
                        toast.show();}
                    }
                else{
                    Toast toast =
                            Toast.makeText(
                                    getApplicationContext(), "Nome o password non inseriti", Toast.LENGTH_SHORT);
                    toast.show();
                }
            }
        });

    }
    private Map<String, String> verificateData(String nomeUtente, String password) throws JSONException, IOException {
        JSONArray ds=datasetParser.readDataset(getBaseContext());
        int i=0;
        Map<String, String> m = new HashMap<>();
        while(i<ds.length()){
            JSONObject utente=(JSONObject) ds.get(i);
            System.out.println(utente.get("name").toString());
            System.out.println(utente.get("password").toString());
            if(((String)utente.get("name")).equals(nomeUtente)){
                if(((String)utente.get("password")).equals(password)){
                    m.put("seed", (String)utente.get("seed"));
                    m.put("azureProfileId", (String)utente.get("azureProfileId"));
                    m.put("hash", (String)utente.get("hash"));
                    return m;
                }
                else return  m;
            }
            i++;
        }
        return m;
    }

}



