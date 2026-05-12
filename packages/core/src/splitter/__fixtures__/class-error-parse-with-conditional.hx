package haxe.io;

class BytesBuffer extends BaseBuffer implements IBuffer {
	#if neko
	var b:Dynamic;
	#elseif flash
	var b:ByteArray;
	#end
}
