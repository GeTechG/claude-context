package haxe.io;

@:forward(copy, length)
abstract Bytes(BytesData) from Array<UInt8> to BytesData {
	public var length(default, null):Int;
	function new(length, b) {
		this.length = length;
	}
}
