package com.example.biometricsystem;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.widget.SectionIndexer;

import java.io.Serializable;
import java.util.List;


public interface SimilarityClassifier  {

    void register(String name, Recognition recognition);

    List<Recognition> recognizeImage(Bitmap bitmap, boolean getExtra);

    void enableStatLogging(final boolean debug);

    String getStatString();

    void close();

    void setNumThreads(int num_threads);

    void setUseNNAPI(boolean isChecked);

    /** Risultato restituito da un classifier, contenente il risultato della recognition */
    public class Recognition implements Serializable{
        /**
         * Identificatore univoco che identifica il contenuto della recognition.
         */
        private final String id;

        /** Nome della recognition */
        private  String title;

        /**
         *Score che quantifica quanto è buona la recognition in relazione alle altre.Più bassa è meglio.
         */
        private final Float distance;
        private Object extra;

        /**Locazione opzionale della locazione della source dentro l'immagine responsabile della riconoscimento.*/
        private RectF location;
        private Integer color;
        private Bitmap crop;
        private Bitmap rgba;

        public Recognition(
                final String id, final String title, final Float distance, final RectF location) {
            this.id = id;
            this.title = title;
            this.distance = distance;
            this.location = location;
            this.color = null;
            this.extra = null;
            this.crop = null;
            this.rgba=null;
        }

        public void setExtra(Object extra) {
            this.extra = extra;
        }
        public Object getExtra() {
            return this.extra;
        }

        public void setColor(Integer color) {
            this.color = color;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getDistance() {
            return distance;
        }

        public RectF getLocation() {
            return new RectF(location);
        }
        public void setTitle(String title){this.title=title;}
        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (distance != null) {
                resultString += String.format("(%.1f%%) ", distance * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }

        public Integer getColor() {
            return this.color;
        }

        public void setCrop(Bitmap crop) {
            this.crop = crop;
        }
        public void setRgba(Bitmap rgba){this.rgba=rgba;}
        public Bitmap getCrop() {
            return this.crop;
        }
        public Bitmap getRgba(){return rgba;}
    }
}