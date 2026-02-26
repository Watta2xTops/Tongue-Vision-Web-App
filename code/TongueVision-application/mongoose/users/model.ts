import mongoose, { model } from "mongoose";
import { UserSchema, UserType } from "@/mongoose/users/schema";

// "User" is the model name (capitalized), and it maps to "users" collection in MongoDB.
const UserModel = mongoose.models.User || model<UserType>("User", UserSchema);
export default UserModel;
