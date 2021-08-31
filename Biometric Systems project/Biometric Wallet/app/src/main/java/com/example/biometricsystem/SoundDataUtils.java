package com.example.biometricsystem;

import java.io.DataInputStream;
        import java.io.File;
        import java.io.FileInputStream;
        import java.io.IOException;
        import java.io.InputStream;

class SoundDataUtils {
    public static double[] load16BitPCMRawDataFileAsDoubleArray(File file) {
        InputStream in = null;
        if (file.isFile()) {
            long size = file.length();
            try {
                in = new FileInputStream(file);
                return readStreamAsDoubleArray(in, size);
            } catch (Exception e) {
            }
        }
        return null;
    }

    public static double[] readStreamAsDoubleArray(InputStream in, long size)
            throws IOException {
        int bufferSize = (int) (size / 2);
        double[] result = new double[bufferSize];
        DataInputStream is = new DataInputStream(in);
        for (int i = 0; i < bufferSize; i++) {
            result[i] = is.readShort() / 32768.0;
        }
        return result;
    }

}
