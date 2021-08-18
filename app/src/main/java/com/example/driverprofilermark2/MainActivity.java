package com.example.driverprofilermark2;

import androidx.appcompat.app.AppCompatActivity;

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

import com.example.driverprofilermark2.ml.FullDataDriverprofiler;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements SensorEventListener{

    private static final String TAG = "MainActivity";

    // Sensor declaration
    private SensorManager sensorManager;
    Sensor accelerometer;
    Sensor gyroscope;

    TextView xAcc, yAcc, zAcc, xGyro, yGyro, zGyro;

    Interpreter tflite;

    //Prepare Array list to store the input data
    private static List<Float> ax, ay, az;
    private static List<Float> gx, gy, gz;

    //declare result variable to store array of result;
    private float[] results;

    // TIME STAMP
    private static final int TIME_STAMP = 50;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // initialising variables
        xAcc = (TextView)findViewById(R.id.xAccel);
        yAcc = (TextView)findViewById(R.id.yAccel);
        zAcc = (TextView)findViewById(R.id.zAccel);

        xGyro = (TextView)findViewById(R.id.xGyro);
        yGyro = (TextView)findViewById(R.id.yGyro);
        zGyro = (TextView)findViewById(R.id.zGyro);

        // Initializing the lists for accelerometer and gyroscope
        ax = new ArrayList<>(); ay = new ArrayList<>(); az = new ArrayList<>();
        gx = new ArrayList<>(); gy = new ArrayList<>(); gz = new ArrayList<>();



        Log.d(TAG, "onCreate: Initializing Sensor services");
        // declare sensor manager and get sensors
        sensorManager = (SensorManager)getSystemService(Context.SENSOR_SERVICE);


        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        if(accelerometer != null){
            // Register Listener for accelerometer
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST);
            Log.d(TAG, "onCreate: Accelerometer Initialized");

        }else{

            Log.d(TAG, "onCreate: Accelerometer not supported");

        }

        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        if (gyroscope != null) {
            // Register gyroscope listener
            sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST);
            Log.d(TAG, "onCreate: Gyroscope initialized");

        }else{
            Log.d(TAG, "onCreate: Gyroscope not supported");
        }




    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor sensorType = sensorEvent.sensor;

        if(sensorType.getType()==Sensor.TYPE_ACCELEROMETER) {
            xAcc.setText("xAccel: " + sensorEvent.values[0]);
            yAcc.setText("yAccel: " + sensorEvent.values[1]);
            zAcc.setText("zAccel: " + sensorEvent.values[2]);

            // adding the accelerometer values inside the list
            ax.add(sensorEvent.values[0]);
            ay.add(sensorEvent.values[1]);
            az.add(sensorEvent.values[2]);

            Log.d(TAG, "onSensorChanged: Accelerometer-X: " + ax );
            Log.d(TAG, "onSensorChanged: Accelerometer-Y: " + ay );
            Log.d(TAG, "onSensorChanged: Accelerometer-Z: " + az );

        }else if(sensorType.getType()==Sensor.TYPE_GYROSCOPE){

            xGyro.setText("xAccel: " + sensorEvent.values[0]);
            yGyro.setText("yAccel: " + sensorEvent.values[1]);
            zGyro.setText("zAccel: " + sensorEvent.values[2]);

            // adding the gyroscope values inside the list
            gx.add(sensorEvent.values[0]);
            gy.add(sensorEvent.values[1]);
            gz.add(sensorEvent.values[2]);

           // Log.d(TAG, "onSensorChanged: Gyroscope Sensor values" + gx + "/n" + gy + "/n" + gz);
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
         && gx.size() >= TIME_STAMP && gy.size() >= TIME_STAMP && gz.size() >= TIME_STAMP)
        {
            data.addAll(ax.subList(0, TIME_STAMP));
            data.addAll(ay.subList(0, TIME_STAMP));
            data.addAll(az.subList(0, TIME_STAMP));

            data.addAll(gx.subList(0, TIME_STAMP));
            data.addAll(gy.subList(0, TIME_STAMP));
            data.addAll(gz.subList(0, TIME_STAMP));
        }
        Log.d(TAG, "predictActivities: Data in List ArrayList"+ data);

        try {
            FullDataDriverprofiler model = FullDataDriverprofiler.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 50, 6}, DataType.FLOAT32);

            // creating the input Tensor using the variable 'data'
            TensorBuffer tensorBuffer = TensorBuffer.createDynamic(DataType.FLOAT32);
            tensorBuffer.loadArray(toFloatArray(data));


            ByteBuffer byteBuffer = tensorBuffer.getBuffer();
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            FullDataDriverprofiler.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            Log.d(TAG, "predictActivities: output array: " + outputFeature0);

            // Releases model resources if no longer used.
            model.close();

            //clear the list for the next prediction
            ax.clear(); ay.clear(); az.clear();
            gx.clear(); gy.clear(); gz.clear();


        } catch (IOException e) {
            // TODO Handle the exception
        }

    }

    private float[] toFloatArray(List<Float> data){
        int i = 0;
        float[] array = new float[data.size()];
        for (Float f: data){
            array[i++] = (f !=null ? f: Float.NaN);
        }
        return array;
    }


    @Override
    protected void onResume() {
        super.onResume();

    }

    @Override
    protected void onPause() {
        super.onPause();
    }
}