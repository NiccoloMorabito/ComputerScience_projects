package com.example.biometricsystem.model;

class InnerError {
    public String code;
    public String message;
}

public class AzureApiError {
    public InnerError error;
}
