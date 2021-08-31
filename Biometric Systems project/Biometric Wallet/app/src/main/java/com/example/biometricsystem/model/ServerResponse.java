package com.example.biometricsystem.model;

import com.google.gson.annotations.SerializedName;

public class ServerResponse {

        @SerializedName("success")
        public Boolean success;
        public String user;

        public ServerResponse(Boolean success, String user) {
            this.success = success;
            this.user = user;
        }

}
