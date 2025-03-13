# SoftWhisper March 2025 is out!

**Big Changes Around**
<br> <br>
In the previous release, I was unhappy with the performance and accessibility of our application. Our previous implementation was too heavily reliant on CUDA. 
AMD users would have to install a specific Pytorch package, but it was too difficult to install, and did not provide much of a benefit anyway.

After some research, I created a ZLUDA branch to mimic CUDA; unfortunately, none of the ZLUDA implementations support Pytorch. 

But not all hope was lost.

After more research and frustration, I heard of Whisper.cpp. It is a reimplementation of the OpenAI Whisper API in pure C++, and has no standalone dependencies. 
Since it can easily use Vulkan, combines CPU + GPU acceleration and can be easily compiled on Linux, it would be worth a shot.

The results are very surprising: Whisper.cpp can transcribe 2 hours of audio in around 2-3 minutes with my current hardware. 
By comparison,with multiprocessing and the regular Whisper API, 20-30 minutes of audio will take you around 40 minutes.

Aligning with my design philosophy that software should be as uncomplicated as possible to use, I'm making available a 64-bit pre-compiled version of Whisper.cpp
that supports Vulkan, so all you need to do this time around, if you use Windows, is to download our repository and run the main script with Python:

> Python SoftWhisper.py

And that's it! The models will also be downloaded for you if you don't have them.

Please note that I haven't tested this application under Linux; however, just placing a compiled Whisper.cpp of your choice under the same folder
as the project should work. The default name the application will look for is Whisper_lin-x64; however, you can also select the directory of your choice
by simply starting the application and changing the directory under the option "Whisper.cpp executable."

**Known bugs**

- Despite being very performant, this software still has many more lines of code than it should, which I will probably address in the future. 
- I couldn't get speaker identification to work properly on this release, so it was disabled and removed from the interface.
- When you select a new video, it won't load the video right away. You will need to press play.
