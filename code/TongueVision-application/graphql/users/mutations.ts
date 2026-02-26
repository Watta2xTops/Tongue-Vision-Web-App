import * as userService from "mongoose/users/services";

export const userMutations = {
  createUser: (_: any, args: any) => userService.createUser(args),
  updateUser: (_: any, { userId, ...update }: any) =>
    userService.updateUser(userId, update),
  deleteUser: (_: any, { userId }: { userId: string }) =>
    userService.deleteUser(userId),
};
