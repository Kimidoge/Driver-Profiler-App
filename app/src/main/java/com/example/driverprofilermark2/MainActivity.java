package com.example.driverprofilermark2;

import androidx.appcompat.app.AppCompatActivity;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.Tensor;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;


import com.example.driverprofilermark2.ml.FullDataDriverprofilerNoscaleBatchnormalization;
import com.example.driverprofilermark2.ml.FullDataDriverprofilerNoscaleBatchnormalization300;
import com.example.driverprofilermark2.ml.FullDataDriverprofilerNoscaleDummy2;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.DoubleAccumulator;

public class MainActivity extends AppCompatActivity implements SensorEventListener{

    private static final String TAG = "MainActivity";

    // Sensor declaration
    private SensorManager sensorManager;
    Sensor accelerometer;
    Sensor gyroscope;
    private SensorEventListener accelerometerListener, gyroscopeListener;

    TextView xAcc, yAcc, zAcc, xGyro, yGyro, zGyro;
    TextView textAccelerate, textAgroAccelerate, textAgroBrake, textAgroLeft, textAgroRight,textBrake, textIdling, textLeft, textRight;


    Interpreter tflite;

    //Prepare Array list to store the input data
    private static List<Float> ax, ay, az;
    private static List<Float> gx, gy, gz;

    //declare result variable to store array of result;
    private float[] results;

    // TIME STAMP
    private static final int TIME_STAMP = 300;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // initialising variables
        xAcc = (TextView)findViewById(R.id.xAcc);
        yAcc = (TextView)findViewById(R.id.yAcc);
        zAcc = (TextView)findViewById(R.id.zAcc);

        xGyro = (TextView)findViewById(R.id.xGyro);
        yGyro = (TextView)findViewById(R.id.yGyro);
        zGyro = (TextView)findViewById(R.id.zGyro);

        // Initializing the lists for accelerometer and gyroscope
        ax = new ArrayList<>(); ay = new ArrayList<>(); az = new ArrayList<>();
        gx = new ArrayList<>(); gy = new ArrayList<>(); gz = new ArrayList<>();

        textAccelerate = (TextView) findViewById(R.id.textAccelerate);
        textAgroAccelerate = (TextView) findViewById(R.id.textAgroAcc);
        textAgroBrake = (TextView) findViewById(R.id.textAgroBrake);
        textAgroLeft = (TextView) findViewById(R.id.textAgroLeft);
        textAgroRight = (TextView) findViewById(R.id.textAgroRight);
        textBrake = (TextView) findViewById(R.id.textBrake);
        textIdling = (TextView) findViewById(R.id.textIdling);
        textLeft = (TextView) findViewById(R.id.textLeft);
        textRight = (TextView) findViewById(R.id.textRight);



        Log.d(TAG, "onCreate: Initializing Sensor services");
        // declare sensor manager and get sensors
        sensorManager = (SensorManager)getSystemService(Context.SENSOR_SERVICE);


        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        if(accelerometer != null){
            // Register Listener for accelerometer
            sensorManager.registerListener(this, accelerometer, 10000);
            Log.d(TAG, "onCreate: Accelerometer Initialized");

        }else{

            Log.d(TAG, "onCreate: Accelerometer not supported");

        }

        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        if (gyroscope != null) {
            // Register gyroscope listener
            sensorManager.registerListener(this, gyroscope,10000);
            Log.d(TAG, "onCreate: Gyroscope initialized");

        }else{
            Log.d(TAG, "onCreate: Gyroscope not supported");
        }




    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor sensorType = sensorEvent.sensor;

        if(sensorType.getType()==Sensor.TYPE_ACCELEROMETER) {

            // adding the accelerometer values inside the list
            ax.add(sensorEvent.values[0]);
            ay.add(sensorEvent.values[1]);
            az.add(sensorEvent.values[2]);

            xAcc.setText("X-ACC: "+ sensorEvent.values[0]);
            yAcc.setText("Y-ACC: "+ sensorEvent.values[1]);
            zAcc.setText("Z-ACC: "+ sensorEvent.values[2]);


           Log.d(TAG, "onSensorChanged: List Ax: " + ax );
           Log.d(TAG, "onSensorChanged: List Ay: " + ay );
           Log.d(TAG, "onSensorChanged: List Az: " + az );

        }else if(sensorType.getType()==Sensor.TYPE_GYROSCOPE){

            // adding the gyroscope values inside the list
            gx.add(sensorEvent.values[0]);
            gy.add(sensorEvent.values[1]);
            gz.add(sensorEvent.values[2]);

           // Log.d(TAG, "onSensorChanged: Gyroscope Sensor values" + gx + "/n" + gy + "/n" + gz);

            xGyro.setText("X-GYRO: "+ sensorEvent.values[0]);
            yGyro.setText("Y-GYRO: "+ sensorEvent.values[1]);
            zGyro.setText("Z-GYRO: "+ sensorEvent.values[2]);

        }


        predictActivities();

    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    private void predictActivities() {
        //create new list to combine all arrayLists into one big list. Data will be our input variable
        List<Float> data = new ArrayList<>();
        if( ax.size() >= TIME_STAMP && ay.size() >= TIME_STAMP && az.size() >= TIME_STAMP
         && gx.size() >= TIME_STAMP && gy.size() >= TIME_STAMP && gz.size() >= TIME_STAMP) {

            data.addAll(ax.subList(0, TIME_STAMP));
            data.addAll(ay.subList(0, TIME_STAMP));
            data.addAll(az.subList(0, TIME_STAMP));

            data.addAll(gx.subList(0, TIME_STAMP));
            data.addAll(gy.subList(0, TIME_STAMP));
            data.addAll(gz.subList(0, TIME_STAMP));

           Log.d(TAG, "predictActivities: Data in data (combined)" + data);  // manually counted, input shape is 50 time-steps of 6 features ========

           // float[] input = toFloatArray(data);  //<===============CULPRIT - DOES NOT CONVERT PROPERLY INTO FLOAT ARRAY

            try {


                FullDataDriverprofilerNoscaleDummy2 model = FullDataDriverprofilerNoscaleDummy2.newInstance(getApplicationContext());
                float[] input = toFloatArray(data);
                //Log.d(TAG, "predictActivities: toFloatArray: " +  input.);

                // Creates inputs for reference.
                TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 300, 6}, DataType.FLOAT32);

                // creating the TensorBuffer for inputting the float array
                //TensorBuffer tensorBuffer = TensorBuffer.createDynamic(DataType.FLOAT32);
                //tensorBuffer.loadArray(toFloatArray(data));

                // ByteBuffer byteBuffer = tensorBuffer.getBuffer();
                inputFeature0.loadArray(toFloatArray(data));


                // Runs model inference and gets result.
                FullDataDriverprofilerNoscaleDummy2.Outputs outputs = model.process(inputFeature0);



                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                textAccelerate.setText("Accelerate: \t" + outputFeature0.getFloatArray()[0]) ;
                textAgroAccelerate.setText("Aggressive Accelerate: \t" + outputFeature0.getFloatArray()[1]);
                textAgroBrake.setText("Aggressive Brake: \t" + outputFeature0.getFloatArray()[2]);
                textAgroLeft.setText("Aggressive Left: \t" + outputFeature0.getFloatArray()[3]);
                textAgroRight.setText("Aggressive Right: \t" + outputFeature0.getFloatArray()[4] );
                textBrake.setText("Brake: \t" + outputFeature0.getFloatArray()[5] );
                textIdling.setText("Idling: \t" + outputFeature0.getFloatArray()[6] );
                textLeft.setText("Left: \t" +outputFeature0.getFloatArray()[7] );
                textRight.setText("Right: \t" + outputFeature0.getFloatArray()[8] );

                Log.d(TAG, "predictActivities: output array: "
                        +outputFeature0.getFloatArray()[0]+ "\t\t" + outputFeature0.getFloatArray()[1] + "\t\t"
                        +outputFeature0.getFloatArray()[2] + "\t\t" + outputFeature0.getFloatArray()[3] + "\t\t"
                        +outputFeature0.getFloatArray()[4] + "\t\t" + outputFeature0.getFloatArray()[5] + "\t\t"
                        + outputFeature0.getFloatArray()[6] + "\t\t" + outputFeature0.getFloatArray()[7] + "\t\t"
                        + outputFeature0.getFloatArray()[8]
                );

                //clear the list for the next prediction

                // Releases model resources if no longer used.
                model.close();
                data.clear();
                ax.clear();
                ay.clear();
                az.clear();
                gx.clear();
                gy.clear();
                gz.clear();


            } catch (IOException e) {
                // TODO Handle the exception
                e.printStackTrace();
            }
        }

            }


    private float[] toFloatArray(List<Float> data){
       int i = 0;

        float[] array = new float[data.size()];
        for (Float f: data){
            array[i++] = (f !=null ? f: Float.NaN);
        }
        //Log.d(TAG, "toFloatArray: " + array);
        return array;


    }

    private float Round(float value, int decimal_places){
        BigDecimal bigDecimal = new BigDecimal(Float.toString(value));
        bigDecimal = bigDecimal.setScale(decimal_places, BigDecimal.ROUND_HALF_UP);
        return bigDecimal.floatValue();
    }


    @Override
    protected void onResume() {
        super.onResume();

        sensorManager.registerListener(this, gyroscope, 10000);
        sensorManager.registerListener(this, accelerometer, 10000);
        Toast.makeText(this, "onResume started", Toast.LENGTH_SHORT).show();

    }

    @Override
    protected void onPause() {
        super.onPause();

        Toast.makeText(this, "onPause started", Toast.LENGTH_SHORT).show();
        sensorManager.unregisterListener(accelerometerListener);
        sensorManager.unregisterListener(gyroscopeListener);
        super.onPause();
    }
}