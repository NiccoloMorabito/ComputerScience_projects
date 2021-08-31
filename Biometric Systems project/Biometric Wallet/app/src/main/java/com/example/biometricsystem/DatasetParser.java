package com.example.biometricsystem;

import android.content.Context;
import org.json.JSONArray;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;


public class DatasetParser {
    final private String FILE_NAME="utenti.json";
    public void insertDataset(JSONObject utente, Context context) throws IOException,JSONException{

    File file=new File(context.getFilesDir(),FILE_NAME);
    FileReader fileReader=null;
    FileWriter fileWriter=null;
    BufferedReader bufferedReader=null;
    BufferedWriter bufferdWriter=null;
    String response=null;

        try{
            if(!file.exists()){
            file.createNewFile();
                JSONArray ds=new JSONArray();
                ds.put(utente);
                fileWriter=new FileWriter(file.getAbsoluteFile());
                bufferdWriter=new BufferedWriter(fileWriter);
                bufferdWriter.write(ds.toString());
                bufferdWriter.close();
            }
            else{
            JSONArray ds=readDataset(context);
            ds.put(utente);
            fileWriter=new FileWriter(file.getAbsoluteFile());
            bufferdWriter=new BufferedWriter(fileWriter);
            bufferdWriter.write(ds.toString());
            bufferdWriter.close();}
           }

        catch (IOException e){e.printStackTrace();}





}
public JSONArray readDataset(Context context) throws IOException,JSONException{
    File file=new File(context.getFilesDir(),FILE_NAME);
    FileReader fileReader=null;
    BufferedReader bufferedReader=null;
    String response=null;
    StringBuffer output=new StringBuffer();
    fileReader=new FileReader(file.getAbsolutePath());
    bufferedReader=new BufferedReader(fileReader);
    String line="";
    while((line=bufferedReader.readLine())!=null){
        output.append(line+'\n');
    }
    response=output.toString();
    JSONArray fine=new JSONArray(response);
    bufferedReader.close();
    return fine;

}
public JSONObject parseUtente(String name,String id,float distance,Integer color,float top,
                              float bottom, float right, float left,byte[] rgba,byte[] gray,
                              String password,float[][] extra,String Seed, String azureProfileId, String hash) throws JSONException {
//First Employee
    JSONObject utente = new JSONObject();
    utente.put("name",name);
    utente.put("password",password);
    utente.put("id",id);
    utente.put("distance",distance);
    utente.put("color",color);
    utente.put("top",top);
    utente.put("bottom",bottom);
    utente.put("let",left);
    utente.put("right",right);
    utente.put("hash",hash);
    JSONArray grayA= new JSONArray();
    for(byte b: gray){
        grayA.put(b);
    }
    JSONArray rgbaA= new JSONArray();
    for(byte b: rgba){
        rgbaA.put(b);
    }
    utente.put("rgba",rgbaA);
    utente.put("grey",grayA);
    utente.put("seed",Seed);
    utente.put("azureProfileId",azureProfileId);
   // JSONArray jsonArray = new JSONArray();
   // for (float[] ca : extra) {
     //   JSONArray arr = new JSONArray();
      //  for (float c : ca) {
        //    arr.put(c); // or some other conversion
       // }
       // jsonArray.put(arr);
   // }


    utente.put("extra",Arrays.deepToString(extra));
    return utente;

}
}
