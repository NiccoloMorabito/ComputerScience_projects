package com.example.biometricsystem;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

public class Registrazione extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_registrazione);


        Button go=(Button) findViewById(R.id.go_enroll);
        go.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String  nome=((EditText)findViewById(R.id.registrazione_nome)).getText().toString();
                String password=((EditText)findViewById(R.id.registrazione_password)).getText().toString();
                System.out.println("nome:"+nome+";");
                System.out.println("password:"+password+";");

                if(!nome.equals("")&&!password.equals("")){
                Intent intent = new Intent(getBaseContext(), Enrollment.class);
                intent.putExtra("nome",nome).putExtra("password",password);
                startActivity(intent);
                finish();}
                else{
                    Toast toast =
                            Toast.makeText(
                                    getApplicationContext(), "Nome o password non inseriti", Toast.LENGTH_SHORT);
                    toast.show();
                }

            }
        });
    }

}